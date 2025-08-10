#!/usr/bin/env python3
"""
LLM-Enhanced Consensus Agents

This module provides LLM-enhanced agents that can be used in consensus experiments,
based on the SimpleConsensusAgent pattern but integrated with the existing
FaultTolerantAgent framework.
"""

import json
import time
import re
import random
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import the existing agent framework
from advanced_fault_tolerant_consensus import FaultTolerantAgent, HPCJob, HPCNode

# Try to import LLM providers, fall back to mock if unavailable
try:
    import sys
    sys.path.append('.')
    from sambanova_consensus_demo import SambaNova_LLMManager
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  SambaNova LLM not available, using mock LLM for experiments")

class MockLLMManager:
    """Mock LLM manager for testing when real LLM is unavailable"""
    
    def __init__(self):
        self.response_patterns = {
            "proposal": [
                '{"node_id": "reliable_node", "score": 0.9, "reason": "optimal resource match"}',
                '{"node_id": "high_memory_node", "score": 0.85, "reason": "memory intensive workload"}',
                '{"node_id": "gpu_node", "score": 0.95, "reason": "GPU acceleration needed"}',
            ],
            "vote": [
                '{"vote": "accept", "confidence": 0.8}',
                '{"vote": "accept", "confidence": 0.9}',
                '{"vote": "reject", "confidence": 0.7}',
            ]
        }
    
    def query(self, prompt: str, query_type: str, temperature: float = 0.0, max_tokens: int = 100) -> str:
        """Generate a mock response based on query type"""
        time.sleep(random.uniform(0.5, 2.0))  # Simulate LLM thinking time
        
        if query_type in self.response_patterns:
            return random.choice(self.response_patterns[query_type])
        else:
            return '{"error": "unknown query type"}'

class LLMEnhancedFaultTolerantAgent(FaultTolerantAgent):
    """Fault-tolerant agent with LLM decision-making capabilities"""
    
    def __init__(self, agent_id: str, stake: int, specialization: str = "general", 
                 llm_temperature: float = 0.0, llm_max_tokens: int = 100):
        super().__init__(agent_id, stake)
        self.specialization = specialization
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        
        # Initialize LLM manager
        if LLM_AVAILABLE:
            self.llm_manager = SambaNova_LLMManager()
        else:
            self.llm_manager = MockLLMManager()
        
        # Performance tracking
        self.llm_call_count = 0
        self.llm_total_time = 0.0
        self.llm_success_rate = 0.0
    
    def llm_proposal(self, job: HPCJob, available_nodes: List[HPCNode]) -> Dict:
        """Create a proposal using LLM reasoning"""
        
        # Prepare context for LLM
        job_context = f"""
Job Requirements:
- Name: {job.name}
- Nodes needed: {job.nodes_required}
- CPU per node: {job.cpu_per_node}
- Memory per node: {job.memory_per_node}GB
- GPU per node: {job.gpu_per_node}
- Priority: {job.priority}
- Job type: {job.job_type}
"""

        # Prepare available nodes context
        nodes_context = "Available Nodes:\n"
        for node in available_nodes[:5]:  # Limit to first 5 for prompt efficiency
            nodes_context += f"- {node.id}: {node.cpu_cores}CPU/{node.memory_gb}GB/{node.gpu_count}GPU ({node.node_type})\n"
        
        # Agent specialization context
        specialization_context = f"Your specialization: {self.specialization} specialist"
        
        # Create focused prompt
        prompt = f"""{job_context}
{nodes_context}
{specialization_context}

As a {self.specialization} specialist, analyze this job and recommend the best node allocation.
Consider resource requirements, node capabilities, and your expertise.

Respond ONLY with valid JSON:
{{"node_id": "best_node_id", "score": 0.95, "reasoning": "brief explanation"}}"""

        try:
            start_time = time.time()
            self.llm_call_count += 1
            
            response = self.llm_manager.query(
                prompt, "proposal", 
                temperature=self.llm_temperature, 
                max_tokens=self.llm_max_tokens
            )
            
            elapsed_time = time.time() - start_time
            self.llm_total_time += elapsed_time
            
            result = self._extract_json_from_response(response)
            
            if result and "node_id" in result:
                return result
            else:
                return self._fallback_proposal(job, available_nodes)
                
        except Exception as e:
            print(f"    ❌ LLM proposal failed for {self.agent_id}: {e}")
            return self._fallback_proposal(job, available_nodes)
    
    def llm_vote(self, job: HPCJob, proposed_allocation: Dict) -> Dict:
        """Vote on a proposal using LLM reasoning"""
        
        proposal_context = f"""
Proposal to Vote On:
- Job: {job.name} ({job.job_type})
- Proposed nodes: {proposed_allocation.get('nodes', 'unknown')}
- Proposer reasoning: {proposed_allocation.get('reasoning', 'not provided')}
"""

        agent_context = f"Your role: {self.specialization} specialist with stake {self.stake}"
        
        prompt = f"""{proposal_context}
{agent_context}

As a {self.specialization} specialist, evaluate this resource allocation proposal.
Consider if it matches the job requirements and aligns with your expertise.

Respond ONLY with valid JSON:
{{"vote": "accept", "confidence": 0.85, "reasoning": "brief explanation"}}"""

        try:
            start_time = time.time()
            self.llm_call_count += 1
            
            response = self.llm_manager.query(
                prompt, "vote",
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens
            )
            
            elapsed_time = time.time() - start_time
            self.llm_total_time += elapsed_time
            
            result = self._extract_json_from_response(response)
            
            if result and "vote" in result:
                return result
            else:
                return self._fallback_vote(job, proposed_allocation)
                
        except Exception as e:
            print(f"    ❌ LLM vote failed for {self.agent_id}: {e}")
            return self._fallback_vote(job, proposed_allocation)
    
    def _extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON from LLM response using the pattern from SimpleConsensusAgent"""
        if not response:
            return {}
        
        # Find first JSON-like structure
        json_start = response.find('{')
        if json_start == -1:
            return {}
        
        # Find matching closing brace
        brace_count = 0
        json_end = json_start
        for i, char in enumerate(response[json_start:], json_start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        try:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback extraction using regex
            return self._manual_extraction(response)
    
    def _manual_extraction(self, text: str) -> Dict:
        """Manual extraction for common patterns"""
        result = {}
        
        # Extract vote
        vote_match = re.search(r'"vote":\s*"(accept|reject)"', text, re.IGNORECASE)
        if vote_match:
            result["vote"] = vote_match.group(1).lower()
        
        # Extract node_id
        node_match = re.search(r'"node_id":\s*"([^"]+)"', text, re.IGNORECASE)
        if node_match:
            result["node_id"] = node_match.group(1)
        
        # Extract score/confidence
        score_match = re.search(r'"(?:score|confidence)":\s*([0-9.]+)', text)
        if score_match:
            key = "score" if "score" in text else "confidence"
            result[key] = float(score_match.group(1))
        
        # Extract reasoning
        reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', text, re.IGNORECASE)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1)
        
        return result if result else {"vote": "accept", "confidence": 0.5}
    
    def _fallback_proposal(self, job: HPCJob, available_nodes: List[HPCNode]) -> Dict:
        """Intelligent fallback proposal based on specialization"""
        best_node = available_nodes[0] if available_nodes else None
        score = 0.5
        reasoning = f"Fallback {self.specialization} choice"
        
        if not best_node:
            return {"error": "no available nodes"}
        
        # Apply specialization-based heuristics
        if self.specialization == "gpu" and available_nodes:
            for node in available_nodes:
                if node.gpu_count >= job.gpu_per_node:
                    best_node = node
                    score = 0.85
                    reasoning = "GPU specialist found suitable GPU node"
                    break
        
        elif self.specialization == "memory" and available_nodes:
            for node in available_nodes:
                if node.memory_gb >= job.memory_per_node * 1.5:  # 50% memory buffer
                    best_node = node
                    score = 0.8
                    reasoning = "Memory specialist selected high-memory node"
                    break
        
        elif self.specialization == "compute" and available_nodes:
            for node in available_nodes:
                if node.cpu_cores >= job.cpu_per_node:
                    best_node = node
                    score = 0.75
                    reasoning = "Compute specialist found adequate CPU resources"
                    break
        
        return {
            "node_id": best_node.id,
            "score": score,
            "reasoning": reasoning
        }
    
    def _fallback_vote(self, job: HPCJob, proposed_allocation: Dict) -> Dict:
        """Simple fallback voting logic"""
        # Basic heuristic: accept if it seems reasonable for our specialization
        confidence = 0.6
        
        if self.specialization in proposed_allocation.get("reasoning", "").lower():
            vote = "accept"
            confidence = 0.75
            reasoning = f"Aligns with {self.specialization} expertise"
        else:
            vote = "accept" if random.random() > 0.3 else "reject"
            reasoning = f"General {self.specialization} assessment"
        
        return {
            "vote": vote,
            "confidence": confidence,
            "reasoning": reasoning
        }
    
    def get_llm_performance_stats(self) -> Dict:
        """Get LLM performance statistics"""
        avg_time = self.llm_total_time / max(1, self.llm_call_count)
        
        return {
            "llm_calls": self.llm_call_count,
            "total_llm_time": self.llm_total_time,
            "avg_llm_time": avg_time,
            "specialization": self.specialization
        }

def create_llm_enhanced_agent(agent_id: str, stake: int, 
                            specialization: str = "general",
                            llm_temperature: float = 0.0,
                            llm_max_tokens: int = 100) -> LLMEnhancedFaultTolerantAgent:
    """Factory function to create LLM-enhanced agents"""
    return LLMEnhancedFaultTolerantAgent(
        agent_id=agent_id,
        stake=stake,
        specialization=specialization,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens
    )

def create_hybrid_agent_pool(num_agents: int, 
                           heuristic_fraction: float = 0.5,
                           llm_temperature: float = 0.0,
                           llm_max_tokens: int = 100) -> List[FaultTolerantAgent]:
    """Create a mixed pool of heuristic and LLM agents"""
    
    agents = []
    specializations = ["gpu", "memory", "compute", "network", "storage", "general"]
    agent_names = [
        "PRIMARY_CONTROLLER", "GPU_CLUSTER_MANAGER", "CPU_CLUSTER_MANAGER",
        "MEMORY_MANAGER", "STORAGE_MANAGER", "BACKUP_CONTROLLER", 
        "EDGE_COORDINATOR", "SECONDARY_CONTROLLER", "NETWORK_MANAGER"
    ]
    
    num_heuristic = int(num_agents * heuristic_fraction)
    num_llm = num_agents - num_heuristic
    
    # Create heuristic agents
    for i in range(num_heuristic):
        agent_name = agent_names[i] if i < len(agent_names) else f"HEURISTIC_AGENT_{i}"
        stake = random.randint(100, 500)
        agent = FaultTolerantAgent(agent_name, stake)
        agent.agent_type = "heuristic"
        agents.append(agent)
    
    # Create LLM agents
    for i in range(num_llm):
        agent_idx = num_heuristic + i
        agent_name = agent_names[agent_idx] if agent_idx < len(agent_names) else f"LLM_AGENT_{agent_idx}"
        stake = random.randint(100, 500)
        specialization = specializations[i % len(specializations)]
        
        agent = create_llm_enhanced_agent(
            agent_id=agent_name,
            stake=stake,
            specialization=specialization,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens
        )
        agent.agent_type = "llm"
        agents.append(agent)
    
    return agents

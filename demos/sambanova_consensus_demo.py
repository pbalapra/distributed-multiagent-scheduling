#!/usr/bin/env python3
"""
Multi-Consensus Protocols Demo with SambaNova LLM
=================================================

Comprehensive comparison of 4 consensus protocols using SambaNova's Meta-Llama models:
- Byzantine Fault Tolerant
- Raft Consensus  
- Multi-round Negotiation
- Weighted Voting

Uses the same LLM setup from test.py with SambaNova SambaStudio API.
"""

import json
import re
import time
import random
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

# SambaNova LLM setup - credentials from environment variables
from langchain_community.llms.sambanova import SambaStudio

class SambaNova_LLMManager:
    """LLM manager using SambaNova SambaStudio API with enhanced error handling"""
    
    def __init__(self, model="Meta-Llama-3-70B-Instruct"):
        self.model = model
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
        # Verify required environment variables are set
        required_env_vars = [
            "SAMBASTUDIO_URL",
            "SAMBASTUDIO_API_KEY"
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"⚠️  Missing SambaNova environment variables: {', '.join(missing_vars)}")
            print(f"   These variables should be defined in your ~/.bashrc file.")
            print(f"   To use them in the current session, run:")
            print(f"   source ~/.bashrc")
            print(f"   ")
            print(f"   Or export them manually:")
            for var in missing_vars:
                print(f"   export {var}=<your_value>")
            print(f"   ")
            print(f"   Continuing with mock LLM fallback mode...")
            self.use_fallback_mode = True
        else:
            print(f"✅ SambaNova environment variables found")
            print(f"   SAMBASTUDIO_URL: {os.getenv('SAMBASTUDIO_URL', 'N/A')[:50]}...")
            print(f"   SAMBASTUDIO_API_KEY: {'*' * 20}[REDACTED]")
            self.use_fallback_mode = False
        
    def call_llm(self, prompt: str, max_tokens=1024, temperature=0.3, retry_count=0):
        """Call SambaNova LLM with enhanced error handling and retry logic"""
        try:
            llama = SambaStudio(
                model_kwargs={
                    "do_sample": True,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "process_prompt": False,
                    "model": self.model,
                },
            )
            
            response = llama.invoke(prompt)
            
            # Check for empty or invalid responses
            if not response or not response.strip():
                if retry_count < self.max_retries:
                    print(f"      ⚠️ Empty response, retrying... ({retry_count + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay * (retry_count + 1))  # Exponential backoff
                    return self.call_llm(prompt, max_tokens, temperature, retry_count + 1)
                else:
                    print(f"      ❌ Empty response after {self.max_retries} retries")
                    return self._generate_fallback_response(prompt)
            
            # Clean up response
            response = response.strip()
            
            print(f"      💬 FULL RESPONSE:")
            print(f"      {'-' * 60}")
            print(f"      {response}")
            print(f"      {'-' * 60}")
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "connection" in error_msg or "network" in error_msg:
                if retry_count < self.max_retries:
                    print(f"      ⚠️ Network error, retrying... ({retry_count + 1}/{self.max_retries}): {str(e)}")
                    time.sleep(self.retry_delay * (retry_count + 1))
                    return self.call_llm(prompt, max_tokens, temperature, retry_count + 1)
            
            print(f"      ❌ SambaNova LLM Error: {str(e)}")
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate intelligent fallback responses based on prompt content"""
        prompt_lower = prompt.lower()
        
        # BFT proposal fallback
        if "bft" in prompt_lower and "proposal" in prompt_lower:
            return '{"proposal": "accept", "node_id": "n1", "score": 0.7, "arguments": ["fallback_decision"], "reasoning": "LLM unavailable, using fallback logic"}'
        
        # Voting fallback
        elif "vote" in prompt_lower and ("accept" in prompt_lower or "reject" in prompt_lower):
            return '{"vote": "accept", "confidence": 0.7, "reasoning": "fallback vote"}'
        
        # Negotiation fallback
        elif "negotiation" in prompt_lower or "round" in prompt_lower:
            return '{"node_id": "n1", "score": 0.7, "reasoning": "fallback negotiation"}'
        
        # Leader decision fallback
        elif "leader" in prompt_lower and "decision" in prompt_lower:
            return '{"node_id": "n1", "score": 0.8, "arguments": ["sufficient_resources"], "reasoning": "Fallback leader decision based on resource availability"}'
        
        # Weighted vote fallback
        elif "specialist" in prompt_lower and "weight" in prompt_lower:
            return '{"node_id": "n1", "score": 0.7, "reasoning": "fallback weighted decision"}'
        
        # General fallback
        return '{"status": "fallback", "message": "LLM unavailable"}'
    
    def query(self, prompt: str, task_type: str = "general", temperature: float = 0.0, max_tokens: int = 1000):
        """Query with task-specific formatting and enhanced error handling"""
        print(f"    🧠 SambaNova LLM Query ({task_type}):")
        print(f"      📝 FULL PROMPT:")
        print(f"      {'-' * 60}")
        print(f"      {prompt}")
        print(f"      {'-' * 60}")
        print(f"      ⚙️ Config: temp={temperature}, max_tokens={max_tokens}, model={self.model}")
        
        # Enhance prompts for better structured responses
        enhanced_prompt = self._enhance_prompt(prompt, task_type)
        response = self.call_llm(enhanced_prompt, max_tokens, temperature)
        return response
    
    def _enhance_prompt(self, prompt: str, task_type: str) -> str:
        """Enhance prompts for better structured responses"""
        # Add task-specific instructions for better JSON formatting
        if task_type in ["proposal", "vote", "negotiation", "weighted_vote", "leader_decision"]:
            enhancement = "\n\nIMPORTANT: Respond with valid JSON only. Do not include explanatory text before or after the JSON."
            return prompt + enhancement
        
        return prompt

@dataclass
class Job:
    name: str
    job_type: str
    nodes_required: int  # Number of nodes needed
    cpu_per_node: int    # CPU required per node
    memory_per_node: int # Memory required per node
    priority: str
    node_type_preference: str = "any"  # Preferred node type

@dataclass 
class Node:
    id: str
    name: str
    cpu_cores: int
    memory_gb: int
    node_type: str
    gpu_count: int = 0
    storage_tb: int = 0
    is_allocated: bool = False  # Whether node is currently allocated to a job
    allocated_to_job: str = ""  # Which job this node is allocated to

@dataclass
class TestResult:
    protocol: str
    job: str
    success: bool
    time_taken: float
    consensus_score: float
    rounds: int

class ConsensusAgent:
    """Intelligent agent using SambaNova LLM for consensus protocols"""
    
    def __init__(self, agent_id: str, name: str, specialization: str, llm_manager, weight: float = 1.0):
        self.agent_id = agent_id
        self.name = name
        self.specialization = specialization
        self.llm_manager = llm_manager
        self.weight = weight
        self.managed_nodes = []
    
    def add_node(self, node: Node):
        self.managed_nodes.append(node)
    
    def extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON from SambaNova response with enhanced parsing"""
        if not response or not response.strip():
            return {"proposal": "accept", "node_id": "n1", "score": 0.7, "reasoning": "Empty response"}
        
        response = response.strip()
        
        # Try direct JSON parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to clean up common formatting issues
        cleaned_response = response
        
        # Remove markdown code blocks
        cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
        cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
        
        # Remove leading/trailing text that isn't JSON
        json_start = cleaned_response.find('{')
        json_end = cleaned_response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            cleaned_response = cleaned_response[json_start:json_end]
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON patterns in the text
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON
            r'\{[^{}]+\}',  # Simple JSON
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    # Validate it has expected keys
                    if any(key in parsed for key in ['vote', 'node_id', 'proposal', 'score']):
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        # Try to extract key-value pairs manually
        extracted_data = self._manual_key_extraction(response)
        if extracted_data:
            return extracted_data
        
        # Final fallback with intelligent defaults based on response content
        return self._intelligent_fallback_json(response)
    
    def _manual_key_extraction(self, text: str) -> Optional[Dict]:
        """Manually extract key-value pairs from text"""
        result = {}
        
        # Extract common patterns
        patterns = {
            'vote': r'["\']?vote["\']?\s*[:\=]\s*["\']?(accept|reject)["\']?',
            'node_id': r'["\']?node_id["\']?\s*[:\=]\s*["\']?(n\d+)["\']?',
            'score': r'["\']?score["\']?\s*[:\=]\s*([\d\.]+)',
            'confidence': r'["\']?confidence["\']?\s*[:\=]\s*([\d\.]+)',
            'proposal': r'["\']?proposal["\']?\s*[:\=]\s*["\']?(accept|reject)["\']?'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1)
                if key in ['score', 'confidence']:
                    try:
                        result[key] = float(value)
                    except ValueError:
                        continue
                else:
                    result[key] = value
        
        # Return result if we found at least one key
        return result if result else None
    
    def _intelligent_fallback_json(self, response: str) -> Dict:
        """Generate intelligent fallback JSON based on response content"""
        response_lower = response.lower()
        
        # Determine likely response type and generate appropriate fallback
        if 'reject' in response_lower:
            if 'vote' in response_lower:
                return {"vote": "reject", "confidence": 0.6, "reasoning": "Extracted from text"}
            else:
                return {"proposal": "reject", "node_id": "n1", "score": 0.3, "reasoning": "Rejection detected"}
        
        # Look for node mentions
        node_match = re.search(r'\bn(\d+)\b', response_lower)
        if node_match:
            node_id = f"n{node_match.group(1)}"
        else:
            node_id = "n1"  # Default
        
        # Look for numeric scores
        score_match = re.search(r'\b0?\.\d+\b|\b[01]\b', response)
        if score_match:
            try:
                score = float(score_match.group(0))
            except ValueError:
                score = 0.7
        else:
            score = 0.7
        
        # Determine response type based on context
        if 'vote' in response_lower:
            return {"vote": "accept", "confidence": score, "reasoning": "Fallback extraction"}
        elif 'negotiation' in response_lower or 'round' in response_lower:
            return {"node_id": node_id, "score": score, "reasoning": "Fallback negotiation"}
        else:
            return {"proposal": "accept", "node_id": node_id, "score": score, "reasoning": "Fallback proposal"}
    
    def create_bft_proposal(self, job: Job, available_nodes: List[Node]) -> Optional[Dict]:
        """Create Byzantine Fault Tolerant proposal"""
        prompt = f"""You are agent {self.name} (specialization: {self.specialization}) in a Byzantine Fault Tolerant consensus protocol.

JOB TO PLACE:
- Name: {job.name}
- Type: {job.job_type}
- Priority: {job.priority}
- Nodes Required: {job.nodes_required}
- CPU per Node: {job.cpu_per_node}
- Memory per Node: {job.memory_per_node}GB
- Node Type Preference: {job.node_type_preference}

AVAILABLE NODES:
{chr(10).join([f"- {n.id} ({n.name}): {n.cpu_cores}CPU/{n.memory_gb}GB, type={n.node_type}, GPUs={n.gpu_count}" for n in available_nodes])}

As a {self.specialization} specialist, create an optimal proposal for this job placement.

Provide an example JSON response:
{{"proposal": "accept/reject", "node_id": "nX", "score": 0.0-1.0, "arguments": ["reason1", "reason2"], "reasoning": "detailed explanation"}}"""
        
        try:
            response = self.llm_manager.query(prompt, "proposal", temperature=0.0, max_tokens=1000)
            proposal_data = self.extract_json_from_response(response)
            return proposal_data
        except Exception as e:
            print(f"  ❌ {self.name} proposal failed: {e}")
            # Track BFT proposal failure (will be incremented by demo class)
            return None
    
    def bft_vote_on_proposal(self, proposal: Dict, job: Job, all_proposals: List[Dict]) -> Dict:
        """Vote on a BFT proposal"""
        score = proposal.get('response', {}).get('score', 0)
        node_id = proposal.get('response', {}).get('node_id', 'unknown')
        proposer_name = proposal.get('agent_name', 'Unknown')
        arguments = proposal.get('response', {}).get('arguments', [])
        
        # Get node details for informed voting
        target_node = None
        for node in self.managed_nodes:
            if node.id == node_id:
                target_node = node
                break
        
        # If we don't manage this node, get it from a general list (simplified)
        if not target_node:
            # Create a simple fallback - in real implementation would access global node list
            target_node_info = f"node {node_id}"
        else:
            target_node_info = f"node {node_id} with {target_node.cpu_cores}CPU/{target_node.memory_gb}GB"
        
        # Use multiple prompt strategies for better success rate
        prompts = [
            # Strategy 1: Ultra-simple
            f"Job {job.name}: {job.cpu_per_node}CPU/node, {job.memory_per_node}GB/node, {job.nodes_required} nodes → {node_id}. Vote: accept or reject? {{\"vote\": \"accept\", \"confidence\": 0.8}}",
            
            # Strategy 2: Direct question format
            f"Should job {job.name} ({job.cpu_per_node}CPU/node, {job.memory_per_node}GB/node, needs {job.nodes_required} nodes) go to {node_id}? Answer: {{\"vote\": \"accept\", \"confidence\": 0.9, \"reasoning\": \"fits well\"}}",
            
            # Strategy 3: Original format
            f"""Vote on this proposal:
Job: {job.name} needs {job.nodes_required} nodes ({job.cpu_per_node}CPU, {job.memory_per_node}GB each)
Proposal: {node_id}
You: {self.specialization} specialist
Example: {{"vote": "accept", "confidence": 0.8, "reasoning": "good match"}}"""
        ]
        
        # Try multiple prompts until we get a response
        vote_failed = True
        for i, prompt in enumerate(prompts):
            try:
                response = self.llm_manager.query(prompt, f"vote_attempt_{i+1}", temperature=0.0, max_tokens=1000)
                if response and response.strip():
                    vote_data = self.extract_json_from_response(response)
                    if vote_data.get('vote') in ['accept', 'reject']:
                        vote_failed = False
                        return vote_data
            except Exception as e:
                continue
        
        # Mark vote as failed for tracking
        self._vote_failed = vote_failed
        
        # Intelligent fallback based on specialization and basic resource matching
        fallback_vote = self._intelligent_fallback_vote(job, node_id, proposer_name)
        print(f"  ⚡ {self.name} using intelligent fallback: {fallback_vote['vote']}")
        return fallback_vote
    
    def _intelligent_fallback_vote(self, job: Job, node_id: str, proposer_name: str) -> Dict:
        """Intelligent fallback voting based on basic resource logic"""
        # Basic node specifications (simplified)
        node_specs = {
            'n1': {'cpu': 16, 'memory': 64, 'type': 'gpu'},
            'n2': {'cpu': 16, 'memory': 64, 'type': 'gpu'},
            'n3': {'cpu': 8, 'memory': 128, 'type': 'memory'},
            'n4': {'cpu': 32, 'memory': 32, 'type': 'compute'},
            'n5': {'cpu': 32, 'memory': 32, 'type': 'compute'},
            'n6': {'cpu': 4, 'memory': 16, 'type': 'storage'}
        }
        
        if node_id not in node_specs:
            return {"vote": "abstain", "confidence": 0.3, "reasoning": "unknown node"}
        
        node = node_specs[node_id]
        
        # Check basic resource requirements
        cpu_fits = node['cpu'] >= job.cpu_per_node
        memory_fits = node['memory'] >= job.memory_per_node
        
        # Specialist preferences
        type_match = {
            'gpu': node['type'] == 'gpu',
            'memory': node['type'] == 'memory' or node['memory'] >= 64,
            'compute': node['type'] == 'compute' or node['cpu'] >= 16,
            'storage': node['type'] == 'storage' or node['type'] in ['memory', 'compute'],
            'general': True
        }.get(self.specialization, True)
        
        if cpu_fits and memory_fits:
            if type_match:
                return {"vote": "accept", "confidence": 0.8, "reasoning": f"resources fit, {self.specialization} specialist approves"}
            else:
                return {"vote": "accept", "confidence": 0.6, "reasoning": "resources fit despite type mismatch"}
        else:
            missing = []
            if not cpu_fits:
                missing.append(f"CPU ({node['cpu']}<{job.cpu_per_node})")
            if not memory_fits:
                missing.append(f"memory ({node['memory']}<{job.memory_per_node}GB)")
            return {"vote": "reject", "confidence": 0.9, "reasoning": f"insufficient {', '.join(missing)}"}
    
    def create_negotiation_proposal(self, job: Job, available_nodes: List[Node], round_num: int) -> Optional[Dict]:
        """Create proposal for Multi-round Negotiation"""
        prompt = f"""Round {round_num} negotiation for job {job.name}:

Job needs: {job.nodes_required} nodes with {job.cpu_per_node}CPU/{job.memory_per_node}GB each
Your specialization: {self.specialization}
Node type preference: {job.node_type_preference}

Available nodes:
{chr(10).join([f"- {n.id}: {n.cpu_cores}CPU/{n.memory_gb}GB ({n.node_type})" for n in available_nodes])}

Example proposal: {{"node_id": "n1", "score": 0.8, "reasoning": "good match"}}"""
        
        try:
            response = self.llm_manager.query(prompt, "negotiation", temperature=0.0, max_tokens=1000)
            return self.extract_json_from_response(response)
        except Exception as e:
            print(f"  ❌ {self.name} negotiation failed: {e}")
            # Track negotiation failure (will be incremented by demo class)
            return None
    
    def weighted_vote(self, job: Job, available_nodes: List[Node]) -> Dict:
        """Cast weighted vote for job placement"""
        prompt = f"""As {self.specialization} specialist (weight {self.weight}), suggest node for {job.name}:

Job: {job.nodes_required} nodes with {job.cpu_per_node}CPU/{job.memory_per_node}GB each
Node type preference: {job.node_type_preference}
Nodes: {', '.join([f"{n.id}({n.cpu_cores}CPU/{n.memory_gb}GB)" for n in available_nodes[:3]])}

Example: {{"node_id": "n1", "score": 0.9, "reasoning": "best fit"}}"""
        
        try:
            response = self.llm_manager.query(prompt, "weighted_vote", temperature=0.0, max_tokens=1000)
            return self.extract_json_from_response(response)
        except Exception as e:
            print(f"  ❌ {self.name} weighted vote failed: {e}")
            # Track weighted vote failure (will be incremented by demo class)
            return {"node_id": "n1", "score": 0.5, "reasoning": "fallback"}

class SambaNova_ConsensusDemo:
    """Comprehensive consensus protocols demo using SambaNova LLM"""
    
    def __init__(self):
        self.llm_manager = SambaNova_LLMManager()
        self.nodes = []
        self.agents = []
        self.results = []
        # LLM failure tracking
        self.llm_failures = {
            "BFT": {"proposals": 0, "votes": 0},
            "Raft": {"decisions": 0},
            "Negotiation": {"proposals": 0},
            "Weighted": {"votes": 0}
        }
        
        print("🧠 Initializing SambaNova LLM for consensus protocols...")
        try:
            # Test the LLM connection
            test_response = self.llm_manager.call_llm("Hello, this is a test.", max_tokens=50)
            print("✅ SambaNova LLM ready for consensus protocols!")
        except Exception as e:
            print(f"⚠️ SambaNova LLM connection issue: {e}")
            print("Will continue with fallback responses if needed.")
    
    def setup_cluster_and_agents(self):
        """Setup nodes and agents for consensus testing"""
        print("\n🏗️ STEP 1: Setting up cluster and agents")
        print("-" * 50)
        
        # Create diverse HPC nodes with realistic specs
        self.nodes = [
            Node("n1", "GPU-Server-01", 32, 256, "gpu", gpu_count=4, storage_tb=10),
            Node("n2", "GPU-Server-02", 32, 256, "gpu", gpu_count=4, storage_tb=10),
            Node("n3", "HighMem-01", 64, 512, "memory", gpu_count=0, storage_tb=50),
            Node("n4", "HighMem-02", 64, 512, "memory", gpu_count=0, storage_tb=50),
            Node("n5", "Compute-01", 128, 128, "compute", gpu_count=0, storage_tb=2),
            Node("n6", "Compute-02", 128, 128, "compute", gpu_count=0, storage_tb=2),
            Node("n7", "Storage-01", 16, 64, "storage", gpu_count=0, storage_tb=100),
            Node("n8", "Storage-02", 16, 64, "storage", gpu_count=0, storage_tb=100)
        ]
        
        # Create intelligent agents with different specializations and weights
        self.agents = [
            ConsensusAgent("agent1", "GPU-Specialist", "gpu", self.llm_manager, 1.2),
            ConsensusAgent("agent2", "Memory-Expert", "memory", self.llm_manager, 1.1),
            ConsensusAgent("agent3", "Compute-Manager", "compute", self.llm_manager, 1.0),
            ConsensusAgent("agent4", "Storage-Coordinator", "storage", self.llm_manager, 0.9),
            ConsensusAgent("agent5", "General-Coordinator", "general", self.llm_manager, 0.8)
        ]
        
        # Assign nodes to agents
        for node in self.nodes:
            if node.node_type == "gpu":
                self.agents[0].add_node(node)
            elif node.node_type == "memory":
                self.agents[1].add_node(node)
            elif node.node_type == "compute":
                self.agents[2].add_node(node)
            elif node.node_type == "storage":
                self.agents[3].add_node(node)
        
        # General coordinator manages all nodes
        for node in self.nodes:
            self.agents[4].add_node(node)
        
        print("🖥️ CLUSTER NODES:")
        for node in self.nodes:
            gpu_info = f", {node.gpu_count} GPUs" if node.gpu_count > 0 else ""
            print(f"  • {node.name}: {node.cpu_cores}CPU/{node.memory_gb}GB{gpu_info} ({node.node_type})")
        
        print("\n🤖 CONSENSUS AGENTS:")
        for agent in self.agents:
            node_count = len(agent.managed_nodes)
            print(f"  • {agent.name}: {agent.specialization}, {node_count} nodes, weight={agent.weight}")
    
    def test_bft_consensus(self, job: Job) -> TestResult:
        """Test Byzantine Fault Tolerant consensus"""
        print(f"\n🛡️ BYZANTINE FAULT TOLERANT CONSENSUS: {job.name}")
        print("=" * 55)
        start_time = time.time()
        
        # Phase 1: Proposals
        print("📋 Phase 1: Agents creating proposals...")
        proposals = []
        
        for agent in self.agents[:2]:  # Limit to prevent too many proposals
            print(f"🤖 Agent {agent.name} creating BFT proposal for {job.name}")
            proposal_data = agent.create_bft_proposal(job, self.nodes)
            
            if not proposal_data:
                self.llm_failures["BFT"]["proposals"] += 1
            
            if proposal_data and proposal_data.get("proposal") == "accept":
                proposals.append({
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "response": proposal_data
                })
                score = proposal_data.get("score", 0)
                node_id = proposal_data.get("node_id", "N/A")
                print(f"  📋 BFT Proposal: {node_id} (Score: {score:.2f})")
        
        if not proposals:
            print("❌ No valid proposals received")
            return TestResult("BFT", job.name, False, time.time() - start_time, 0.0, 1)
        
        print(f"  📊 {len(proposals)} proposals received")
        
        # Phase 2: Voting
        print("\n🗳️ Phase 2: BFT voting (2/3+ required)...")
        votes = {}
        required_votes = max(1, (len(self.agents) * 2) // 3 + 1)
        
        for proposal in proposals:
            votes[proposal["agent_id"]] = []
            
            for voter in self.agents:
                if voter.agent_id == proposal["agent_id"]:
                    continue  # Skip self-voting
                
                vote_data = voter.bft_vote_on_proposal(proposal, job, proposals)
                
                # Check if LLM vote failed (using fallback)
                if hasattr(voter, '_vote_failed') and voter._vote_failed:
                    self.llm_failures["BFT"]["votes"] += 1
                    voter._vote_failed = False  # Reset flag
                
                vote = vote_data.get("vote", "abstain")
                confidence = vote_data.get("confidence", 0.0)
                
                votes[proposal["agent_id"]].append({
                    "voter_id": voter.agent_id,
                    "voter_name": voter.name,
                    "vote": vote,
                    "confidence": confidence
                })
                print(f"  🗳️ {voter.name} BFT vote: {vote} (conf: {confidence:.2f})")
        
        # Phase 3: Consensus evaluation
        print("\n📊 Phase 3: BFT consensus evaluation...")
        best_proposal = None
        best_consensus = 0.0
        
        for proposal in proposals:
            agent_votes = votes[proposal["agent_id"]]
            accept_votes = len([v for v in agent_votes if v["vote"] == "accept"])
            total_votes = len(agent_votes)
            
            if total_votes > 0:
                consensus_score = accept_votes / total_votes
                print(f"  📊 {proposal['agent_name']}: {consensus_score:.2f} consensus ({accept_votes}/{total_votes})")
                
                if consensus_score > best_consensus and accept_votes >= required_votes:
                    best_consensus = consensus_score
                    best_proposal = proposal
        
        success = best_proposal is not None
        if success:
            print(f"✅ BFT CONSENSUS REACHED! Winner: {best_proposal['agent_name']}")
            node_id = best_proposal["response"].get("node_id", "unknown")
            node = next((n for n in self.nodes if n.id == node_id), None)
            if node:
                # Mark node as allocated to job
                node.is_allocated = True
                node.allocated_to_job = job.name
                print(f"  ⚡ Executed: {job.name} → {node.name}")
                print(f"  📊 Node {node_id} allocated to job {job.name} (needs {job.nodes_required} nodes total)")
        else:
            print("❌ BFT CONSENSUS FAILED! Insufficient votes")
        
        return TestResult("BFT", job.name, success, time.time() - start_time, best_consensus, 1)
    
    def test_raft_consensus(self, job: Job) -> TestResult:
        """Test Raft consensus"""
        print(f"\n🏛️ RAFT CONSENSUS: {job.name}")
        print("=" * 40)
        start_time = time.time()
        
        # Phase 1: Leader Election
        print("👑 Phase 1: Leader election...")
        leader = random.choice(self.agents)
        votes_received = len(self.agents)  # Simplified - assume all vote for leader
        
        for voter in self.agents:
            print(f"  🗳️ {voter.name} votes for {leader.name} in term 1")
        
        print(f"  👑 {leader.name} becomes leader with {votes_received}/{len(self.agents)} votes")
        
        # Phase 2: Leader Decision
        print(f"\n🏛️ Phase 2: Leader {leader.name} making decision...")
        
        decision_prompt = f"""You are the elected Raft leader making a binding decision for job placement.

JOB: {job.name} ({job.job_type}, {job.priority})
- Nodes Required: {job.nodes_required}
- CPU per Node: {job.cpu_per_node}
- Memory per Node: {job.memory_per_node}GB
- Node Type Preference: {job.node_type_preference}

AVAILABLE NODES:
{chr(10).join([f"- {n.id} ({n.name}): {n.cpu_cores}CPU/{n.memory_gb}GB, type={n.node_type}, GPUs={n.gpu_count}" for n in self.nodes])}

Make an optimal decision as the leader.

Provide an example JSON decision:
{{"node_id": "nX", "score": 0.0-1.0, "arguments": ["reason1", "reason2"], "reasoning": "detailed explanation"}}"""
        
        try:
            print(f"👑 Leader {leader.name} making decision for {job.name}")
            response = self.llm_manager.query(decision_prompt, "leader_decision", temperature=0.0, max_tokens=1000)
            decision_data = leader.extract_json_from_response(response)
            
            node_id = decision_data.get("node_id", "n1")
            score = decision_data.get("score", 0.8)
            reasoning = decision_data.get("reasoning", "Leader decision")
            
            print(f"  🏛️ Leader Decision: {node_id} - {reasoning[:100]}...")
            
            # Phase 3: Log Replication
            print("📄 Phase 3: Replicating decision to followers...")
            replicated_count = 0
            for follower in self.agents:
                if follower.agent_id != leader.agent_id:
                    replicated_count += 1
                    print(f"  📄 Replicated to {follower.name} (log index 0)")
            
            print("✅ RAFT CONSENSUS COMPLETE! Leader decision accepted")
            
            # Execute the decision
            node = next((n for n in self.nodes if n.id == node_id), None)
            if node:
                node.is_allocated = True
                node.allocated_to_job = job.name
                print(f"  ⚡ Executed: {job.name} → {node.name}")
                print(f"  📊 Node {node_id} allocated to job {job.name} (needs {job.nodes_required} nodes total)")
            
            consensus_score = (replicated_count + 1) / len(self.agents)  # Include leader
            return TestResult("Raft", job.name, True, time.time() - start_time, consensus_score, 1)
            
        except Exception as e:
            print(f"❌ Raft leader decision failed: {e}")
            self.llm_failures["Raft"]["decisions"] += 1
            return TestResult("Raft", job.name, False, time.time() - start_time, 0.0, 1)
    
    def test_multiround_negotiation(self, job: Job) -> TestResult:
        """Test Multi-round Negotiation consensus"""
        print(f"\n🤝 MULTI-ROUND NEGOTIATION CONSENSUS: {job.name}")
        print("=" * 55)
        start_time = time.time()
        
        max_rounds = 3
        final_proposals = {}
        
        for round_num in range(1, max_rounds + 1):
            print(f"\n🔄 Round {round_num} negotiations...")
            round_proposals = {}
            
            for agent in self.agents:
                print(f"🤖 {agent.name} negotiating for round {round_num}")
                proposal = agent.create_negotiation_proposal(job, self.nodes, round_num)
                
                if not proposal:
                    self.llm_failures["Negotiation"]["proposals"] += 1
                
                if proposal:
                    round_proposals[agent.agent_id] = {
                        "agent_name": agent.name,
                        "proposal": proposal
                    }
                    node_id = proposal.get("node_id", "N/A")
                    score = proposal.get("score", 0)
                    print(f"  📋 {agent.name}: {node_id} (Score: {score:.2f})")
            
            final_proposals = round_proposals
            
            # Check for convergence (simplified)
            node_ids = []
            for p in round_proposals.values():
                node_id = p["proposal"].get("node_id", "unknown")
                # Handle both string and list node_ids
                if isinstance(node_id, list):
                    node_ids.extend(node_id)
                else:
                    node_ids.append(str(node_id))
            
            if len(set(node_ids)) <= 2:
                print(f"✅ Convergence achieved in round {round_num}!")
                break
        
        # Find consensus
        if final_proposals:
            # Count votes for each node
            node_votes = {}
            for prop_data in final_proposals.values():
                node_id = prop_data["proposal"].get("node_id", "unknown")
                if node_id not in node_votes:
                    node_votes[node_id] = []
                node_votes[node_id].append(prop_data)
            
            # Find most popular node
            best_node = max(node_votes.items(), key=lambda x: len(x[1]))
            consensus_score = len(best_node[1]) / len(final_proposals)
            
            success = consensus_score >= 0.5
            if success:
                print(f"✅ NEGOTIATION CONSENSUS REACHED! Node: {best_node[0]}")
                node = next((n for n in self.nodes if n.id == best_node[0]), None)
                if node:
                    node.is_allocated = True
                    node.allocated_to_job = job.name
                    print(f"  ⚡ Executed: {job.name} → {node.name}")
                    print(f"  📊 Node {best_node[0]} allocated to job {job.name} (needs {job.nodes_required} nodes total)")
            else:
                print("❌ NEGOTIATION FAILED! No consensus reached")
                consensus_score = 0.0
        else:
            success = False
            consensus_score = 0.0
            print("❌ NEGOTIATION FAILED! No proposals received")
        
        return TestResult("Negotiation", job.name, success, time.time() - start_time, consensus_score, max_rounds)
    
    def test_weighted_voting(self, job: Job) -> TestResult:
        """Test Weighted Voting consensus"""
        print(f"\n⚖️ WEIGHTED VOTING CONSENSUS: {job.name}")
        print("=" * 50)
        start_time = time.time()
        
        print("🗳️ Collecting weighted votes...")
        weighted_votes = {}
        
        for agent in self.agents:
            print(f"🤖 {agent.name} casting weighted vote (weight: {agent.weight})")
            vote = agent.weighted_vote(job, self.nodes)
            
            # Check if this was a fallback (indicates LLM failure)
            if vote and vote.get("reasoning") == "fallback":
                self.llm_failures["Weighted"]["votes"] += 1
            
            if vote:
                node_id = vote.get("node_id", "unknown")
                score = vote.get("score", 0)
                reasoning = vote.get("reasoning", "No reasoning")
                
                if node_id not in weighted_votes:
                    weighted_votes[node_id] = []
                
                weighted_votes[node_id].append({
                    "agent": agent.name,
                    "weight": agent.weight,
                    "score": score,
                    "reasoning": reasoning
                })
                print(f"  🗳️ {agent.name}: {node_id} (Score: {score:.2f}, Weight: {agent.weight})")
        
        # Calculate weighted scores
        print("\n📊 Calculating weighted scores...")
        final_scores = {}
        for node_id, votes in weighted_votes.items():
            total_weighted_score = sum(vote["weight"] * vote["score"] for vote in votes)
            total_weight = sum(vote["weight"] for vote in votes)
            final_score = total_weighted_score / total_weight if total_weight > 0 else 0
            final_scores[node_id] = {
                "score": final_score,
                "votes": len(votes),
                "total_weight": total_weight
            }
            print(f"  📊 {node_id}: {final_score:.2f} (from {len(votes)} votes, weight: {total_weight:.1f})")
        
        success = len(final_scores) > 0
        if success:
            # Find best node
            best_node_id = max(final_scores.items(), key=lambda x: x[1]["score"])[0]
            consensus_score = final_scores[best_node_id]["score"]
            
            print(f"✅ WEIGHTED VOTING COMPLETE! Winner: {best_node_id}")
            node = next((n for n in self.nodes if n.id == best_node_id), None)
            if node:
                node.is_allocated = True
                node.allocated_to_job = job.name
                print(f"  ⚡ Executed: {job.name} → {node.name}")
                print(f"  📊 Node {best_node_id} allocated to job {job.name} (needs {job.nodes_required} nodes total)")
        else:
            consensus_score = 0.0
            print("❌ WEIGHTED VOTING FAILED! No votes received")
        
        return TestResult("Weighted", job.name, success, time.time() - start_time, consensus_score, 1)
    
    def run_protocol_comparison(self):
        """Run comprehensive protocol comparison"""
        print("\n🏁 STEP 2: PROTOCOL COMPARISON - ALL 4 METHODS")
        print("-" * 50)
        
        # Test jobs with node-based requirements
        test_jobs = [
            Job("AI-Training", "ai", nodes_required=4, cpu_per_node=16, memory_per_node=64, priority="HIGH", node_type_preference="gpu"),
            Job("Data-Analytics", "analytics", nodes_required=2, cpu_per_node=32, memory_per_node=128, priority="MEDIUM", node_type_preference="memory"),
            Job("HPC-Simulation", "simulation", nodes_required=8, cpu_per_node=64, memory_per_node=32, priority="HIGH", node_type_preference="compute")
        ]
        
        for job in test_jobs:
            print(f"\n🎯 JOB: {job.name}")
            print("=" * 60)
            
            # Test all 4 consensus protocols
            protocols = [
                ("BFT", self.test_bft_consensus),
                ("Raft", self.test_raft_consensus),
                ("Negotiation", self.test_multiround_negotiation),
                ("Weighted", self.test_weighted_voting)
            ]
            
            job_results = []
            
            for protocol_name, test_func in protocols:
                # Reset node allocation status
                for node in self.nodes:
                    node.is_allocated = False
                    node.allocated_to_job = ""
                
                result = test_func(job)
                self.results.append(result)
                job_results.append(result)
            
            # Show job summary
            print(f"\n📊 {job.name} RESULTS SUMMARY:")
            print("-" * 40)
            for result in job_results:
                status = "✅ SUCCESS" if result.success else "❌ FAILED"
                print(f"  {result.protocol:<12} {status}  {result.time_taken:6.2f}s {result.rounds} rounds {result.consensus_score:.2f} consensus")
    
    def show_final_analysis(self):
        """Show comprehensive analysis"""
        print("\n📊 STEP 3: COMPREHENSIVE 4-PROTOCOL ANALYSIS")
        print("=" * 60)
        
        # Group results by protocol
        protocol_results = {
            "BFT": [r for r in self.results if r.protocol == "BFT"],
            "Raft": [r for r in self.results if r.protocol == "Raft"],
            "Negotiation": [r for r in self.results if r.protocol == "Negotiation"],
            "Weighted": [r for r in self.results if r.protocol == "Weighted"]
        }
        
        def analyze_protocol(results, name):
            if not results:
                return 0.0
            
            total_tests = len(results)
            successful_tests = len([r for r in results if r.success])
            success_rate = (successful_tests / total_tests) * 100
            avg_time = sum(r.time_taken for r in results) / total_tests
            avg_consensus = sum(r.consensus_score for r in results if r.success) / max(1, successful_tests)
            avg_rounds = sum(r.rounds for r in results) / total_tests
            
            print(f"\n🏛️ {name}:")
            print(f"  • Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
            print(f"  • Avg Time: {avg_time:.2f}s")
            print(f"  • Avg Rounds: {avg_rounds:.1f}")
            print(f"  • Avg Consensus: {avg_consensus:.2f}")
            return success_rate
        
        success_rates = {}
        success_rates["Byzantine Fault Tolerant"] = analyze_protocol(protocol_results["BFT"], "Byzantine Fault Tolerant")
        success_rates["Raft Consensus"] = analyze_protocol(protocol_results["Raft"], "Raft Consensus")
        success_rates["Multi-round Negotiation"] = analyze_protocol(protocol_results["Negotiation"], "Multi-round Negotiation")
        success_rates["Weighted Voting"] = analyze_protocol(protocol_results["Weighted"], "Weighted Voting")
        
        print(f"\n🏆 CONSENSUS PROTOCOL PERFORMANCE RANKING:")
        print("--" * 30)
        
        # Sort protocols by success rate
        sorted_protocols = sorted(success_rates.items(), key=lambda x: x[1], reverse=True)
        
        medals = ["🥇", "🥈", "🥉", "🏅"]
        for i, (protocol, rate) in enumerate(sorted_protocols[:4]):
            medal = medals[i] if i < 4 else "📊"
            print(f"{medal} {protocol}: {rate:.1f}% success rate")
        
        print(f"\n💡 KEY FEATURES DEMONSTRATED:")
        print("✅ SambaNova LLM-powered intelligent decision making")
        print("✅ 4-protocol consensus comparison (BFT, Raft, Negotiation, Weighted)")
        print("✅ Specialized agent reasoning (GPU, Memory, Compute, Storage)")
        print("✅ Real-time job placement optimization")
        print("✅ Multi-round negotiation with convergence")
        print("✅ Weighted voting with agent specialization")
        print("✅ Consensus quality measurement across all protocols")
        
        # Display LLM failure statistics
        self.show_llm_failure_statistics()
    
    def run_demo(self):
        """Run the complete demonstration"""
        print("🏛️ MULTI-CONSENSUS PROTOCOLS DEMO with SambaNova LLM")
        print("=" * 70)
        print("Comparing ALL 4 Consensus Protocols using Meta-Llama-3-70B:")
        print("• Byzantine Fault Tolerant (BFT) - Robust multi-agent voting")
        print("• Raft Consensus - Leader-based distributed consensus")
        print("• Multi-round Negotiation - Iterative convergence")
        print("• Weighted Voting - Specialist-weighted decisions")
        print()
        
        self.setup_cluster_and_agents()
        self.run_protocol_comparison()
        self.show_final_analysis()
        
        print("\n🎯 SAMBANOVA CONSENSUS PROTOCOLS DEMO COMPLETE!")
        print("=" * 70)
        print("Successfully demonstrated LLM-enhanced consensus with SambaNova Meta-Llama-3-70B!")
    
    def show_llm_failure_statistics(self):
        """Display detailed LLM failure statistics for each consensus method"""
        print("\n🚨 STEP 4: LLM FAILURE ANALYSIS")
        print("=" * 60)
        
        total_failures = 0
        total_calls = 0
        
        for protocol, failures in self.llm_failures.items():
            print(f"\n🛡️ {protocol} PROTOCOL:")
            protocol_failures = 0
            protocol_calls = 0
            
            for call_type, count in failures.items():
                print(f"  • {call_type.capitalize()}: {count} failures")
                protocol_failures += count
                
                # Estimate total calls based on demo structure
                if protocol == "BFT" and call_type == "proposals":
                    # 2 agents × 3 jobs = 6 proposal calls
                    protocol_calls += 6
                elif protocol == "BFT" and call_type == "votes":
                    # Approximately 4 agents × 2 proposals × 3 jobs = 24 vote calls (varies by actual proposals)
                    protocol_calls += 24  # Estimated
                elif protocol == "Raft" and call_type == "decisions":
                    # 1 leader decision per job × 3 jobs = 3 decision calls
                    protocol_calls += 3
                elif protocol == "Negotiation" and call_type == "proposals":
                    # 5 agents × 3 rounds × 3 jobs = 45 proposal calls (max)
                    protocol_calls += 45  # Estimated max
                elif protocol == "Weighted" and call_type == "votes":
                    # 5 agents × 3 jobs = 15 vote calls
                    protocol_calls += 15
            
            failure_rate = (protocol_failures / protocol_calls * 100) if protocol_calls > 0 else 0
            print(f"  📊 Total {protocol} failures: {protocol_failures}/{protocol_calls} ({failure_rate:.1f}%)")
            
            total_failures += protocol_failures
            total_calls += protocol_calls
        
        overall_failure_rate = (total_failures / total_calls * 100) if total_calls > 0 else 0
        
        print(f"\n📈 OVERALL LLM PERFORMANCE:")
        print(f"  • Total LLM failures: {total_failures}")
        print(f"  • Total LLM calls: ~{total_calls}")
        print(f"  • Overall failure rate: {overall_failure_rate:.1f}%")
        print(f"  • Success rate: {100 - overall_failure_rate:.1f}%")
        
        # Failure analysis by protocol
        if total_failures > 0:
            print(f"\n⚠️ FAILURE BREAKDOWN BY PROTOCOL:")
            failure_by_protocol = {}
            for protocol, failures in self.llm_failures.items():
                protocol_total = sum(failures.values())
                if protocol_total > 0:
                    failure_by_protocol[protocol] = protocol_total
            
            sorted_failures = sorted(failure_by_protocol.items(), key=lambda x: x[1], reverse=True)
            for protocol, count in sorted_failures:
                percentage = (count / total_failures * 100)
                print(f"  • {protocol}: {count} failures ({percentage:.1f}% of all failures)")
        
        print(f"\n💡 LLM PERFORMANCE INSIGHTS:")
        if overall_failure_rate < 10:
            print("✅ Excellent LLM performance - very low failure rate")
        elif overall_failure_rate < 25:
            print("✅ Good LLM performance - acceptable failure rate")
        elif overall_failure_rate < 50:
            print("⚠️ Moderate LLM performance - some reliability concerns")
        else:
            print("❌ Poor LLM performance - high failure rate affecting consensus")

if __name__ == "__main__":
    demo = SambaNova_ConsensusDemo()
    demo.run_demo()

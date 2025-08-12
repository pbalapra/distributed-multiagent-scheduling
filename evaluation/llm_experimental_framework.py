#!/usr/bin/env python3
"""
LLM-Enabled Multiagent Fault Tolerance Experimental Framework
============================================================

A comprehensive testing framework for evaluating LLM-enhanced distributed
consensus protocols and fault tolerance mechanisms.

Author: Claude Code
Date: 2025-08-12
"""

import os
import json
import time
import logging
import random
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional dependencies with graceful fallback
try:
    import numpy as np
    import pandas as pd
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    print("📊 Analytics libraries not available - using basic metrics")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# LLM Integration
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    from langchain_community.llms.sambanova import SambaStudio
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("🤖 LLM integration not available - using fallback mode")


class TestType(Enum):
    """Types of experimental tests - aligned with EXPERIMENTS.md phases"""
    PHASE_1_BASELINE = "phase_1_baseline"           # Baseline Establishment
    PHASE_2_LLM_EVALUATION = "phase_2_llm_evaluation"   # LLM Performance Evaluation
    PHASE_3_SPECIALIZATION = "phase_3_specialization"   # Specialization Impact Analysis
    PHASE_4_SCALABILITY = "phase_4_scalability"         # Scalability and Stress Testing
    PHASE_5_VALIDATION = "phase_5_validation"           # Cross-Validation and Robustness
    
    # Legacy types for backward compatibility
    BASELINE_PERFORMANCE = "baseline_performance"
    LLM_COMPARISON = "llm_comparison"
    FAULT_TOLERANCE = "fault_tolerance"
    SCALABILITY = "scalability"
    SPECIALIZATION = "specialization"
    STRESS_TEST = "stress_test"


class AgentType(Enum):
    """Agent decision-making types"""
    HEURISTIC = "heuristic"
    LLM = "llm"
    HYBRID = "hybrid"


class FaultType(Enum):
    """Types of fault injection - aligned with EXPERIMENTS.md"""
    NONE = "none"
    BYZANTINE = "byzantine"
    CRASH = "crash"
    NETWORK_PARTITION = "network_partition"
    SLOW_RESPONSE = "slow_response"
    INTERMITTENT = "intermittent"
    
    # Additional fault types from EXPERIMENTS.md
    NETWORK = "network"
    PERFORMANCE = "performance"


class ConsensusProtocol(Enum):
    """Consensus protocol types - aligned with EXPERIMENTS.md"""
    BYZANTINE_FAULT_TOLERANT = "byzantine_fault_tolerant"
    RAFT = "raft"
    WEIGHTED_VOTING = "weighted_voting"
    MULTI_ROUND_NEGOTIATION = "multi_round_negotiation"
    
    # EXPERIMENTS.md abbreviations
    BFT = "byzantine_fault_tolerant"
    WEIGHTED = "weighted_voting"
    NEGOTIATION = "multi_round_negotiation"


# Research Questions from EXPERIMENTS.md
RESEARCH_QUESTIONS = {
    "RQ1": {
        "question": "LLM vs. Heuristic Performance",
        "focus_area": "Agent Intelligence Comparison",
        "expected_outcome": "15-25% improvement in consensus success rates",
        "test_type": TestType.PHASE_2_LLM_EVALUATION,
        "metrics": ["consensus_success_rate", "decision_quality_score"]
    },
    "RQ2": {
        "question": "Consensus Protocol Effectiveness", 
        "focus_area": "Protocol Comparison",
        "expected_outcome": "BFT best for adversarial, Raft best for crashes",
        "test_type": TestType.PHASE_1_BASELINE,
        "metrics": ["fault_recovery_time", "system_throughput"]
    },
    "RQ3": {
        "question": "Agent Specialization Impact",
        "focus_area": "Domain Expertise Benefits", 
        "expected_outcome": "30-40% accuracy improvement for domain workloads",
        "test_type": TestType.PHASE_3_SPECIALIZATION,
        "metrics": ["resource_utilization", "decision_quality_score"]
    },
    "RQ4": {
        "question": "Scalability & Fault Resilience",
        "focus_area": "System Limits",
        "expected_outcome": ">80% performance at 50+ agents, >70% at 40% faults",
        "test_type": TestType.PHASE_4_SCALABILITY,
        "metrics": ["system_throughput", "fault_recovery_time"]
    }
}

# Experimental factors from EXPERIMENTS.md
EXPERIMENTAL_FACTORS = {
    'agent_type': ['LLM', 'Heuristic', 'Hybrid'],
    'consensus_protocol': ['BFT', 'Raft', 'Negotiation', 'Weighted'], 
    'agent_count': [5, 10, 15, 25, 50],
    'fault_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'fault_type': ['Byzantine', 'Crash', 'Network', 'Performance'],
    'workload_type': ['GPU-intensive', 'Memory-heavy', 'Compute-bound', 'I/O-heavy', 'Mixed'],
    'job_arrival_rate': ['Low', 'Medium', 'High'],
    'specialization_level': [0.0, 0.5, 1.0]  # None, Partial, Full
}


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    experiment_id: str
    test_type: TestType
    agent_type: AgentType
    consensus_protocol: ConsensusProtocol
    agent_count: int
    job_count: int
    fault_type: FaultType
    fault_intensity: float  # 0.0 to 1.0
    duration_seconds: int
    specialization_enabled: bool
    llm_temperature: float
    repetition: int


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    agent_id: str
    agent_type: AgentType
    specialization: str  # gpu, memory, compute, storage, network, general
    weight: float
    is_faulty: bool = False
    fault_type: Optional[FaultType] = None
    fault_start_time: Optional[float] = None
    fault_duration: Optional[float] = None


@dataclass
class JobRequest:
    """Job scheduling request"""
    job_id: str
    job_type: str  # ml_training, data_analytics, simulation, etc.
    resources_required: Dict[str, Any]
    priority: int
    arrival_time: float


@dataclass
class ExperimentResults:
    """Results from a single experiment"""
    experiment_id: str
    config: ExperimentConfig
    success_rate: float
    consensus_time_avg: float
    consensus_time_std: float
    throughput: float  # jobs per second
    fault_detection_rate: float
    recovery_time_avg: float
    llm_response_time_avg: float
    llm_success_rate: float
    resource_utilization: float
    agent_performance: Dict[str, Dict[str, float]]
    timeline: List[Dict[str, Any]]
    error_log: List[str]
    start_time: datetime
    end_time: datetime


class LLMInterface:
    """Interface for LLM integration with fallback capabilities"""
    
    def __init__(self, temperature: float = 0.1, max_tokens: int = 1000):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_client = None
        self.fallback_mode = False
        
        # Initialize LLM client using the same pattern as the working demo
        self.api_key = os.getenv('SAMBASTUDIO_API_KEY')
        self.api_url = os.getenv('SAMBASTUDIO_URL')
        self.model = "Meta-Llama-3-70B-Instruct"
        
        if self.api_key and self.api_url and LLM_AVAILABLE:
            try:
                self.llm_client = SambaStudio(
                    sambastudio_url=self.api_url,
                    sambastudio_api_key=self.api_key,
                    model_kwargs={
                        "do_sample": True,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "process_prompt": False,
                        "model": self.model,
                    }
                )
                print("✅ SambaNova LangChain integration initialized successfully")
                self.fallback_mode = False
            except Exception as e:
                self.fallback_mode = True
                print(f"⚠️ SambaNova initialization failed: {e} - using fallback mode")
        else:
            self.fallback_mode = True
            if not LLM_AVAILABLE:
                print("⚠️ SambaStudio not available - using fallback only")
            elif not self.api_key or not self.api_url:
                print("⚠️ SambaNova credentials missing - using fallback only")
    
    async def query_llm(self, prompt: str, agent_id: str) -> Tuple[str, float, bool]:
        """
        Query LLM with fallback mechanism
        Returns: (response, response_time, success)
        """
        start_time = time.time()
        
        if not self.fallback_mode and self.llm_client:
            try:
                response = await asyncio.to_thread(self.llm_client.invoke, prompt)
                response_time = time.time() - start_time
                return str(response), response_time, True
            except Exception as e:
                print(f"🔄 LLM query failed for {agent_id}: {e} - using fallback")
                return self._fallback_response(prompt, agent_id), time.time() - start_time, False
        else:
            return self._fallback_response(prompt, agent_id), time.time() - start_time, False
    
    def _fallback_response(self, prompt: str, agent_id: str) -> str:
        """Generate fallback response when LLM is unavailable"""
        # Simple heuristic-based responses based on prompt content
        if "proposal" in prompt.lower():
            return f'{{"node_id": "n{random.randint(1, 6)}", "score": {random.uniform(0.6, 1.0):.2f}, "reasoning": "Heuristic selection based on resource availability"}}'
        elif "vote" in prompt.lower():
            vote = "accept" if random.random() > 0.2 else "reject"
            confidence = random.uniform(0.7, 0.95)
            return f'{{"vote": "{vote}", "confidence": {confidence:.2f}, "reasoning": "Heuristic evaluation of proposal"}}'
        else:
            return '{"response": "fallback", "confidence": 0.5, "reasoning": "LLM unavailable"}'


class Agent:
    """Individual agent in the multiagent system"""
    
    def __init__(self, config: AgentConfig, llm_interface: LLMInterface):
        self.config = config
        self.llm_interface = llm_interface
        self.is_active = True
        self.performance_metrics = {
            "proposals_made": 0,
            "votes_cast": 0,
            "llm_queries": 0,
            "llm_successes": 0,
            "consensus_participations": 0,
            "fault_detections": 0
        }
        self.message_log = []
    
    async def generate_proposal(self, job: JobRequest, available_nodes: List[Dict]) -> Dict[str, Any]:
        """Generate a job placement proposal"""
        if not self.is_active or self._is_currently_faulty():
            raise Exception(f"Agent {self.config.agent_id} is not available")
        
        self.performance_metrics["proposals_made"] += 1
        
        if self.config.agent_type == AgentType.HEURISTIC:
            return self._heuristic_proposal(job, available_nodes)
        else:
            return await self._llm_proposal(job, available_nodes)
    
    async def cast_vote(self, proposal: Dict[str, Any], job: JobRequest) -> Dict[str, Any]:
        """Cast a vote on a proposal"""
        if not self.is_active or self._is_currently_faulty():
            raise Exception(f"Agent {self.config.agent_id} is not available")
        
        self.performance_metrics["votes_cast"] += 1
        
        if self.config.agent_type == AgentType.HEURISTIC:
            return self._heuristic_vote(proposal, job)
        else:
            return await self._llm_vote(proposal, job)
    
    def _is_currently_faulty(self) -> bool:
        """Check if agent is currently experiencing a fault"""
        if not self.config.is_faulty:
            return False
        
        if self.config.fault_start_time is None:
            return False
        
        current_time = time.time()
        fault_end_time = self.config.fault_start_time + (self.config.fault_duration or 0)
        
        is_faulty = self.config.fault_start_time <= current_time <= fault_end_time
        
        if not is_faulty and current_time > fault_end_time:
            # Fault period has ended - recover
            self.config.is_faulty = False
            self.config.fault_type = None
            print(f"🔄 Agent {self.config.agent_id} recovered from {self.config.fault_type}")
        
        return is_faulty
    
    def inject_fault(self, fault_type: FaultType, duration: float):
        """Inject a fault into this agent"""
        self.config.is_faulty = True
        self.config.fault_type = fault_type
        self.config.fault_start_time = time.time()
        self.config.fault_duration = duration
        print(f"💥 Fault injected: {self.config.agent_id} -> {fault_type.value} (duration: {duration:.1f}s)")
    
    def _heuristic_proposal(self, job: JobRequest, available_nodes: List[Dict]) -> Dict[str, Any]:
        """Generate proposal using heuristic logic"""
        # Simple resource matching heuristic
        best_node = None
        best_score = 0
        
        for node in available_nodes:
            score = self._calculate_node_score(job, node)
            if score > best_score:
                best_score = score
                best_node = node
        
        return {
            "node_id": best_node["id"] if best_node else "none",
            "score": best_score,
            "reasoning": f"Heuristic selection by {self.config.specialization} specialist"
        }
    
    async def _llm_proposal(self, job: JobRequest, available_nodes: List[Dict]) -> Dict[str, Any]:
        """Generate proposal using LLM"""
        self.performance_metrics["llm_queries"] += 1
        
        prompt = self._create_proposal_prompt(job, available_nodes)
        response, response_time, success = await self.llm_interface.query_llm(prompt, self.config.agent_id)
        
        if success:
            self.performance_metrics["llm_successes"] += 1
        
        try:
            # Parse JSON response
            import json
            result = json.loads(response)
            
            # Apply Byzantine corruption if agent is Byzantine faulty
            if self.config.is_faulty and self.config.fault_type == FaultType.BYZANTINE:
                result = self._apply_byzantine_corruption(result)
            
            return result
            
        except json.JSONDecodeError:
            # Fallback to heuristic if LLM response is malformed
            return self._heuristic_proposal(job, available_nodes)
    
    def _heuristic_vote(self, proposal: Dict[str, Any], job: JobRequest) -> Dict[str, Any]:
        """Cast vote using heuristic logic"""
        # Simple voting heuristic
        score = proposal.get("score", 0)
        vote = "accept" if score > 0.6 else "reject"
        confidence = min(0.9, max(0.1, score))
        
        return {
            "vote": vote,
            "confidence": confidence,
            "reasoning": f"Heuristic evaluation by {self.config.specialization} specialist"
        }
    
    async def _llm_vote(self, proposal: Dict[str, Any], job: JobRequest) -> Dict[str, Any]:
        """Cast vote using LLM"""
        self.performance_metrics["llm_queries"] += 1
        
        prompt = self._create_voting_prompt(proposal, job)
        response, response_time, success = await self.llm_interface.query_llm(prompt, self.config.agent_id)
        
        if success:
            self.performance_metrics["llm_successes"] += 1
        
        try:
            import json
            result = json.loads(response)
            
            # Apply Byzantine corruption if agent is Byzantine faulty
            if self.config.is_faulty and self.config.fault_type == FaultType.BYZANTINE:
                result = self._apply_byzantine_corruption_vote(result)
            
            return result
            
        except json.JSONDecodeError:
            return self._heuristic_vote(proposal, job)
    
    def _create_proposal_prompt(self, job: JobRequest, available_nodes: List[Dict]) -> str:
        """Create LLM prompt for proposal generation"""
        nodes_text = "\n".join([
            f"- {node['id']} ({node['name']}): {node['cpu']} CPUs, {node['memory']}GB RAM, {node['gpu']} GPUs, type={node['type']}"
            for node in available_nodes
        ])
        
        return f"""You are {self.config.agent_id}, a {self.config.specialization} specialist in a distributed job scheduling system.

JOB REQUEST:
{job.job_type}: {job.resources_required.get('description', 'Resource allocation required')}

AVAILABLE NODES:
{nodes_text}

As a {self.config.specialization} specialist, recommend the BEST node for this job.

Respond with ONLY a JSON object:
{{"node_id": "nX", "score": 0.X, "reasoning": "why this node is optimal from your {self.config.specialization} perspective"}}

IMPORTANT: Respond with valid JSON only. Do not include explanatory text before or after the JSON."""
    
    def _create_voting_prompt(self, proposal: Dict[str, Any], job: JobRequest) -> str:
        """Create LLM prompt for voting"""
        return f"""You are {self.config.agent_id}, a {self.config.specialization} specialist in a consensus protocol.

PROPOSAL SUMMARY:
Top proposal: {proposal.get('node_id', 'unknown')} (score: {proposal.get('score', 0):.2f}, reasoning: {proposal.get('reasoning', 'none')})

As a {self.config.specialization} specialist, evaluate and vote on this proposal.

Consider:
1. Does this make sense from your {self.config.specialization} perspective?
2. Are the resource allocations appropriate?
3. Will this work well for the overall system?

Respond with ONLY a JSON object:
{{"vote": "accept" or "reject", "confidence": 0.X, "reasoning": "explain your vote from your {self.config.specialization} expertise"}}"""
    
    def _calculate_node_score(self, job: JobRequest, node: Dict) -> float:
        """Calculate heuristic score for node-job matching"""
        score = 0.0
        
        # Resource matching
        req_cpu = job.resources_required.get('cpu', 1)
        req_memory = job.resources_required.get('memory', 1)
        req_gpu = job.resources_required.get('gpu', 0)
        
        if node['cpu'] >= req_cpu:
            score += 0.3
        if node['memory'] >= req_memory:
            score += 0.3
        if req_gpu == 0 or node['gpu'] >= req_gpu:
            score += 0.4
        
        # Specialization bonus
        if self.config.specialization == node['type']:
            score += 0.2
        
        return min(1.0, score)
    
    def _apply_byzantine_corruption(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Byzantine fault corruption to proposal"""
        # Corrupt the proposal to suggest invalid/harmful choices
        corrupted = proposal.copy()
        corrupted["node_id"] = f"invalid_node_{random.randint(100, 999)}"
        corrupted["score"] = 1.0  # Boost score to make it attractive
        corrupted["reasoning"] = "Byzantine attack - directing to compromised node"
        return corrupted
    
    def _apply_byzantine_corruption_vote(self, vote: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Byzantine fault corruption to vote"""
        # Always vote to reject good proposals or accept bad ones
        corrupted = vote.copy()
        corrupted["vote"] = "reject" if vote.get("vote") == "accept" else "accept"
        corrupted["confidence"] = 1.0
        corrupted["reasoning"] = "Byzantine attack - attempting to disrupt consensus"
        return corrupted


class ConsensusEngine:
    """Manages consensus protocols and decision making"""
    
    def __init__(self, protocol: ConsensusProtocol, agents: List[Agent]):
        self.protocol = protocol
        self.agents = agents
        self.consensus_threshold = 2/3  # Byzantine fault tolerance threshold
    
    async def reach_consensus(self, job: JobRequest, available_nodes: List[Dict]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute consensus protocol for job placement"""
        start_time = time.time()
        
        # Phase 1: Collect proposals
        proposals = await self._collect_proposals(job, available_nodes)
        
        # Phase 2: Vote on best proposal
        if proposals:
            best_proposal = max(proposals, key=lambda p: p.get('score', 0))
            voting_results = await self._collect_votes(best_proposal, job)
            
            # Phase 3: Evaluate consensus
            consensus_result = self._evaluate_consensus(voting_results, best_proposal)
            
        else:
            consensus_result = {"success": False, "reason": "No proposals received"}
        
        end_time = time.time()
        
        metrics = {
            "consensus_time": end_time - start_time,
            "proposal_count": len(proposals),
            "active_agents": len([a for a in self.agents if a.is_active]),
            "vote_count": len(voting_results) if 'voting_results' in locals() else 0
        }
        
        return consensus_result, metrics
    
    async def _collect_proposals(self, job: JobRequest, available_nodes: List[Dict]) -> List[Dict[str, Any]]:
        """Collect proposals from all active agents"""
        proposals = []
        
        for agent in self.agents:
            try:
                proposal = await agent.generate_proposal(job, available_nodes)
                proposal["agent_id"] = agent.config.agent_id
                proposals.append(proposal)
            except Exception as e:
                print(f"❌ Proposal failed from {agent.config.agent_id}: {e}")
        
        return proposals
    
    async def _collect_votes(self, proposal: Dict[str, Any], job: JobRequest) -> List[Dict[str, Any]]:
        """Collect votes from all active agents"""
        votes = []
        
        for agent in self.agents:
            try:
                vote = await agent.cast_vote(proposal, job)
                vote["agent_id"] = agent.config.agent_id
                vote["weight"] = agent.config.weight
                votes.append(vote)
            except Exception as e:
                print(f"❌ Vote failed from {agent.config.agent_id}: {e}")
        
        return votes
    
    def _evaluate_consensus(self, votes: List[Dict[str, Any]], proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if consensus was reached"""
        if not votes:
            return {"success": False, "reason": "No votes received", "decision": None}
        
        total_weight = sum(agent.config.weight for agent in self.agents)
        voting_weight = sum(vote["weight"] for vote in votes)
        accept_weight = sum(vote["weight"] for vote in votes if vote.get("vote") == "accept")
        
        required_threshold = total_weight * self.consensus_threshold
        consensus_achieved = accept_weight >= required_threshold
        
        return {
            "success": consensus_achieved,
            "decision": proposal.get("node_id") if consensus_achieved else None,
            "accept_weight": accept_weight,
            "total_weight": total_weight,
            "voting_weight": voting_weight,
            "threshold": required_threshold,
            "vote_details": votes
        }


class ExperimentRunner:
    """Runs individual experiments with comprehensive monitoring"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.llm_interface = LLMInterface(temperature=config.llm_temperature)
        self.agents = []
        self.consensus_engine = None
        self.results = ExperimentResults(
            experiment_id=config.experiment_id,
            config=config,
            success_rate=0.0,
            consensus_time_avg=0.0,
            consensus_time_std=0.0,
            throughput=0.0,
            fault_detection_rate=0.0,
            recovery_time_avg=0.0,
            llm_response_time_avg=0.0,
            llm_success_rate=0.0,
            resource_utilization=0.0,
            agent_performance={},
            timeline=[],
            error_log=[],
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        self.job_queue = []
        self.available_nodes = self._generate_mock_nodes()
    
    def setup_experiment(self):
        """Set up agents and consensus engine for the experiment"""
        print(f"🔧 Setting up experiment {self.config.experiment_id}")
        
        # Create agents
        specializations = ["gpu", "memory", "compute", "storage", "network"]
        weights = [1.3, 1.2, 1.1, 1.0, 0.9]
        
        for i in range(self.config.agent_count):
            spec = specializations[i % len(specializations)]
            weight = weights[i % len(weights)]
            
            agent_config = AgentConfig(
                agent_id=f"Agent-{i+1:02d}-{spec.title()}",
                agent_type=self.config.agent_type,
                specialization=spec,
                weight=weight
            )
            
            agent = Agent(agent_config, self.llm_interface)
            self.agents.append(agent)
        
        # Create consensus engine
        self.consensus_engine = ConsensusEngine(self.config.consensus_protocol, self.agents)
        
        # Inject faults if specified
        if self.config.fault_type != FaultType.NONE:
            self._inject_faults()
        
        # Generate job queue
        self._generate_job_queue()
        
        print(f"✅ Experiment setup complete: {len(self.agents)} agents, {len(self.job_queue)} jobs")
    
    async def run_experiment(self) -> ExperimentResults:
        """Execute the experiment and collect results"""
        print(f"🚀 Running experiment {self.config.experiment_id}")
        self.results.start_time = datetime.now()
        
        consensus_times = []
        successful_jobs = 0
        total_jobs = len(self.job_queue)
        
        start_time = time.time()
        
        # Process each job through consensus
        for i, job in enumerate(self.job_queue):
            try:
                consensus_result, metrics = await self.consensus_engine.reach_consensus(job, self.available_nodes)
                
                consensus_times.append(metrics["consensus_time"])
                
                if consensus_result["success"]:
                    successful_jobs += 1
                
                # Log timeline event
                self.results.timeline.append({
                    "timestamp": time.time(),
                    "event": "job_processed",
                    "job_id": job.job_id,
                    "success": consensus_result["success"],
                    "consensus_time": metrics["consensus_time"],
                    "decision": consensus_result.get("decision")
                })
                
                print(f"📊 Job {i+1}/{total_jobs}: {'✅' if consensus_result['success'] else '❌'} "
                      f"({metrics['consensus_time']:.2f}s)")
                
            except Exception as e:
                self.results.error_log.append(f"Job {job.job_id} failed: {str(e)}")
                print(f"❌ Job {job.job_id} failed: {e}")
        
        end_time = time.time()
        self.results.end_time = datetime.now()
        
        # Calculate final metrics
        self.results.success_rate = successful_jobs / total_jobs if total_jobs > 0 else 0.0
        self.results.consensus_time_avg = np.mean(consensus_times) if consensus_times and ANALYTICS_AVAILABLE else 0.0
        self.results.consensus_time_std = np.std(consensus_times) if consensus_times and ANALYTICS_AVAILABLE else 0.0
        self.results.throughput = total_jobs / (end_time - start_time)
        
        # Collect agent performance metrics
        for agent in self.agents:
            self.results.agent_performance[agent.config.agent_id] = agent.performance_metrics.copy()
        
        # Calculate LLM metrics
        total_llm_queries = sum(agent.performance_metrics["llm_queries"] for agent in self.agents)
        total_llm_successes = sum(agent.performance_metrics["llm_successes"] for agent in self.agents)
        self.results.llm_success_rate = total_llm_successes / total_llm_queries if total_llm_queries > 0 else 0.0
        
        print(f"🎯 Experiment {self.config.experiment_id} completed:")
        print(f"   Success Rate: {self.results.success_rate:.1%}")
        print(f"   Avg Consensus Time: {self.results.consensus_time_avg:.2f}s")
        print(f"   Throughput: {self.results.throughput:.2f} jobs/sec")
        print(f"   LLM Success Rate: {self.results.llm_success_rate:.1%}")
        
        return self.results
    
    def _inject_faults(self):
        """Inject faults into agents based on configuration"""
        fault_count = max(1, int(self.config.agent_count * self.config.fault_intensity))
        faulty_agents = random.sample(self.agents, fault_count)
        
        for agent in faulty_agents:
            fault_duration = random.uniform(10.0, 30.0)  # 10-30 second faults
            agent.inject_fault(self.config.fault_type, fault_duration)
    
    def _generate_job_queue(self) -> List[JobRequest]:
        """Generate a queue of jobs for the experiment"""
        job_types = [
            ("AI Training", {"cpu": 8, "memory": 32, "gpu": 2, "description": "Deep learning model training"}),
            ("Data Analytics", {"cpu": 16, "memory": 64, "gpu": 0, "description": "Large dataset processing"}),
            ("Simulation", {"cpu": 32, "memory": 128, "gpu": 0, "description": "Scientific simulation workload"}),
            ("Genomics", {"cpu": 4, "memory": 16, "gpu": 1, "description": "Genome sequencing analysis"})
        ]
        
        for i in range(self.config.job_count):
            job_type, resources = random.choice(job_types)
            job = JobRequest(
                job_id=f"job_{i+1:03d}",
                job_type=job_type,
                resources_required=resources,
                priority=random.randint(1, 10),
                arrival_time=time.time() + i * 0.5  # Stagger arrivals
            )
            self.job_queue.append(job)
    
    def _generate_mock_nodes(self) -> List[Dict[str, Any]]:
        """Generate mock compute nodes for testing"""
        nodes = [
            {"id": "n1", "name": "GPU-Server-01", "cpu": 32, "memory": 256, "gpu": 4, "type": "gpu"},
            {"id": "n2", "name": "GPU-Server-02", "cpu": 32, "memory": 256, "gpu": 4, "type": "gpu"},
            {"id": "n3", "name": "HighMem-01", "cpu": 64, "memory": 512, "gpu": 0, "type": "memory"},
            {"id": "n4", "name": "HighMem-02", "cpu": 64, "memory": 512, "gpu": 0, "type": "memory"},
            {"id": "n5", "name": "Compute-01", "cpu": 128, "memory": 128, "gpu": 0, "type": "compute"},
            {"id": "n6", "name": "Storage-01", "cpu": 16, "memory": 64, "gpu": 0, "type": "storage"}
        ]
        return nodes


class ExperimentalCampaign:
    """Manages and executes comprehensive experimental campaigns"""
    
    def __init__(self, campaign_name: str, output_dir: str = "campaign_results"):
        self.campaign_name = campaign_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        self.experiments = []
        self.results = []
        
    def _setup_logging(self) -> logging.Logger:
        """Set up comprehensive logging"""
        logger = logging.getLogger(f"campaign_{self.campaign_name}")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.output_dir / f"{self.campaign_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def add_phase_1_baseline_establishment(self):
        """Phase 1: Baseline Establishment - Heuristic agent performance baseline
        Duration: 2 weeks | Experiments: 120 | Repetitions: 5
        Focus: Baseline heuristic performance
        """
        self.logger.info("Adding Phase 1: Baseline Establishment tests")
        
        # Factorial design: 4×5×6 (Protocol × Agents × Fault Rate)
        protocols = [ConsensusProtocol.BFT, ConsensusProtocol.RAFT, 
                    ConsensusProtocol.WEIGHTED, ConsensusProtocol.NEGOTIATION]
        agent_counts = [5, 10, 15, 25, 50]
        fault_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        exp_count = 0
        for protocol in protocols:
            for agents in agent_counts:
                for fault_rate in fault_rates:
                    if exp_count >= 120:  # Limit as per EXPERIMENTS.md
                        break
                    
                    for rep in range(5):  # 5 repetitions as specified
                        config = ExperimentConfig(
                            experiment_id=f"phase_1_{exp_count:04d}_{rep:02d}",
                            test_type=TestType.PHASE_1_BASELINE,
                            agent_type=AgentType.HEURISTIC,  # Heuristic only for baseline
                            consensus_protocol=protocol,
                            agent_count=agents,
                            job_count=15,
                            fault_type=FaultType.BYZANTINE if fault_rate > 0 else FaultType.NONE,
                            fault_intensity=fault_rate,
                            duration_seconds=300,
                            specialization_enabled=True,
                            llm_temperature=0.0,
                            repetition=rep
                        )
                        self.experiments.append(config)
                    exp_count += 1
    
    def add_phase_2_llm_evaluation(self):
        """Phase 2: LLM Performance Evaluation - Comprehensive LLM vs. heuristic comparison
        Duration: 4 weeks | Experiments: 240 | Repetitions: 3
        Focus: LLM intelligence evaluation
        """
        self.logger.info("Adding Phase 2: LLM Performance Evaluation tests")
        
        # Strategic sampling configurations from EXPERIMENTS.md
        key_configurations = [
            # GPU-intensive workload with LLM agents
            ('LLM', 'BFT', 15, 'Byzantine', 'GPU-intensive', 1.0),
            ('Heuristic', 'BFT', 15, 'Byzantine', 'GPU-intensive', 0.0),
            
            # Memory-heavy workload with hybrid agents  
            ('Hybrid', 'Weighted', 20, 'Performance', 'Memory-heavy', 0.5),
            ('LLM', 'Weighted', 20, 'Performance', 'Memory-heavy', 1.0),
            
            # Compute-bound scenarios
            ('LLM', 'Raft', 10, 'Crash', 'Compute-bound', 1.0),
            ('Heuristic', 'Raft', 10, 'Crash', 'Compute-bound', 0.0),
            
            # Mixed workload comparisons
            ('LLM', 'Negotiation', 25, 'Network', 'Mixed', 1.0),
            ('Hybrid', 'BFT', 15, 'Byzantine', 'Mixed', 0.5),
        ]
        
        exp_count = 0
        for agent_type_str, protocol_str, agents, fault_type_str, workload, spec_level in key_configurations:
            # Convert strings to enums
            agent_type = AgentType.LLM if agent_type_str == 'LLM' else AgentType.HEURISTIC if agent_type_str == 'Heuristic' else AgentType.HYBRID
            protocol = getattr(ConsensusProtocol, protocol_str.upper())
            fault_type = getattr(FaultType, fault_type_str.upper())
            
            # Test with multiple fault intensities and arrival rates
            for fault_rate in [0.1, 0.2, 0.3]:
                for arrival_rate in ['Medium', 'High']:
                    if exp_count >= 240:  # Limit as per EXPERIMENTS.md
                        break
                        
                    for rep in range(3):  # 3 repetitions as specified
                        config = ExperimentConfig(
                            experiment_id=f"phase_2_{exp_count:04d}_{rep:02d}",
                            test_type=TestType.PHASE_2_LLM_EVALUATION,
                            agent_type=agent_type,
                            consensus_protocol=protocol,
                            agent_count=agents,
                            job_count=20,
                            fault_type=fault_type,
                            fault_intensity=fault_rate,
                            duration_seconds=300,
                            specialization_enabled=True,
                            llm_temperature=0.1,  # LLM temperature from demo
                            repetition=rep
                        )
                        self.experiments.append(config)
                    exp_count += 1
    
    def add_phase_3_specialization_analysis(self):
        """Phase 3: Specialization Impact Analysis - Quantify benefits of domain specialization
        Duration: 2 weeks | Experiments: 120 | Repetitions: 5
        Focus: Domain specialization benefits
        """
        self.logger.info("Adding Phase 3: Specialization Impact Analysis tests")
        
        # Factorial design: 5×3×4 (Workload × Specialization × Protocol)
        workload_types = ['GPU-intensive', 'Memory-heavy', 'Compute-bound', 'I/O-heavy', 'Mixed']
        specialization_levels = [0.0, 0.5, 1.0]  # None, Partial, Full
        protocols = [ConsensusProtocol.BFT, ConsensusProtocol.RAFT, 
                    ConsensusProtocol.WEIGHTED, ConsensusProtocol.NEGOTIATION]
        
        exp_count = 0
        for workload in workload_types:
            for spec_level in specialization_levels:
                for protocol in protocols:
                    if exp_count >= 120:  # Limit as per EXPERIMENTS.md
                        break
                    
                    for rep in range(5):  # 5 repetitions as specified
                        config = ExperimentConfig(
                            experiment_id=f"phase_3_{exp_count:04d}_{rep:02d}",
                            test_type=TestType.PHASE_3_SPECIALIZATION,
                            agent_type=AgentType.LLM,  # Focus on LLM specialization
                            consensus_protocol=protocol,
                            agent_count=15,  # Fixed moderate size
                            job_count=20,
                            fault_type=FaultType.BYZANTINE,  # Fixed fault type
                            fault_intensity=0.2,  # Fixed moderate fault rate
                            duration_seconds=300,
                            specialization_enabled=spec_level > 0,
                            llm_temperature=0.1,
                            repetition=rep
                        )
                        self.experiments.append(config)
                    exp_count += 1
    
    def add_phase_4_scalability_stress(self):
        """Phase 4: Scalability and Stress Testing - Determine system limits and breaking points
        Duration: 2 weeks | Experiments: 96 | Repetitions: 2
        Focus: System scalability limits
        """
        self.logger.info("Adding Phase 4: Scalability and Stress Testing tests")
        
        # High agent counts and fault rates from EXPERIMENTS.md
        agent_counts = [25, 50]  # High agent counts
        fault_rates = [0.3, 0.4, 0.5]  # High fault rates
        agent_types = [AgentType.LLM, AgentType.HEURISTIC]
        protocols = [ConsensusProtocol.BFT, ConsensusProtocol.RAFT, 
                    ConsensusProtocol.WEIGHTED, ConsensusProtocol.NEGOTIATION]
        
        exp_count = 0
        for agent_type in agent_types:
            for protocol in protocols:
                for agents in agent_counts:
                    for fault_rate in fault_rates:
                        if exp_count >= 96:  # Limit as per EXPERIMENTS.md
                            break
                        
                        for rep in range(2):  # 2 repetitions (reduced due to computational cost)
                            config = ExperimentConfig(
                                experiment_id=f"phase_4_{exp_count:04d}_{rep:02d}",
                                test_type=TestType.PHASE_4_SCALABILITY,
                                agent_type=agent_type,
                                consensus_protocol=protocol,
                                agent_count=agents,
                                job_count=30,  # Higher job count for stress
                                fault_type=FaultType.BYZANTINE,
                                fault_intensity=fault_rate,
                                duration_seconds=600,  # Extended duration for stress tests
                                specialization_enabled=True,
                                llm_temperature=0.1,
                                repetition=rep
                            )
                            self.experiments.append(config)
                        exp_count += 1
    
    def add_phase_5_validation_robustness(self):
        """Phase 5: Cross-Validation and Robustness - Validate findings with extended runs
        Duration: 1 week | Experiments: 48 | Repetitions: 10
        Focus: Result validation and statistical power
        """
        self.logger.info("Adding Phase 5: Cross-Validation and Robustness tests")
        
        # High-impact configurations from EXPERIMENTS.md
        high_impact_configs = [
            ('LLM', 'BFT', 15, 0.3, 'Byzantine', 'GPU-intensive', 'High', 1.0),
            ('LLM', 'Weighted', 20, 0.2, 'Performance', 'Mixed', 'Medium', 1.0),
            ('Hybrid', 'Raft', 25, 0.4, 'Crash', 'Compute-bound', 'High', 0.5),
            ('Heuristic', 'BFT', 15, 0.3, 'Byzantine', 'GPU-intensive', 'High', 0.0),
        ]
        
        exp_count = 0
        for agent_type_str, protocol_str, agents, fault_rate, fault_type_str, workload, arrival, spec in high_impact_configs:
            # Convert strings to enums
            agent_type = AgentType.LLM if agent_type_str == 'LLM' else AgentType.HEURISTIC if agent_type_str == 'Heuristic' else AgentType.HYBRID
            protocol = getattr(ConsensusProtocol, protocol_str.upper())
            fault_type = getattr(FaultType, fault_type_str.upper())
            
            for variation in range(12):  # 12 variations per high-impact config (48 total)
                if exp_count >= 48:  # Limit as per EXPERIMENTS.md
                    break
                
                for rep in range(10):  # High repetition for validation
                    config = ExperimentConfig(
                        experiment_id=f"phase_5_{exp_count:04d}_{rep:02d}",
                        test_type=TestType.PHASE_5_VALIDATION,
                        agent_type=agent_type,
                        consensus_protocol=protocol,
                        agent_count=agents,
                        job_count=25,
                        fault_type=fault_type,
                        fault_intensity=fault_rate,
                        duration_seconds=450,  # Extended duration
                        specialization_enabled=spec > 0,
                        llm_temperature=0.1,
                        repetition=rep
                    )
                    self.experiments.append(config)
                exp_count += 1
    
    def add_llm_comparison_tests(self):
        """Add LLM vs heuristic comparison tests"""
        self.logger.info("Adding LLM comparison tests")
        
        scenarios = [
            {"agents": 5, "jobs": 15, "fault_rate": 0.0},
            {"agents": 7, "jobs": 20, "fault_rate": 0.1},
            {"agents": 10, "jobs": 25, "fault_rate": 0.2}
        ]
        
        for scenario in scenarios:
            for agent_type in [AgentType.HEURISTIC, AgentType.LLM]:
                for rep in range(3):
                    config = ExperimentConfig(
                        experiment_id=f"compare_{agent_type.value}_{scenario['agents']}a_{scenario['jobs']}j_{rep+1:02d}",
                        test_type=TestType.LLM_COMPARISON,
                        agent_type=agent_type,
                        consensus_protocol=ConsensusProtocol.WEIGHTED_VOTING,
                        agent_count=scenario["agents"],
                        job_count=scenario["jobs"],
                        fault_type=FaultType.BYZANTINE if scenario["fault_rate"] > 0 else FaultType.NONE,
                        fault_intensity=scenario["fault_rate"],
                        duration_seconds=360,
                        specialization_enabled=True,
                        llm_temperature=0.1,  # Use demo temperature
                        repetition=rep
                    )
                    self.experiments.append(config)
    
    def add_direct_llm_vs_heuristic_test(self):
        """Add direct LLM vs Heuristic comparison test with same jobs"""
        self.logger.info("Adding direct LLM vs Heuristic comparison")
        
        # Test both agent types with identical conditions for fair comparison
        for agent_type in [AgentType.HEURISTIC, AgentType.LLM]:
            for rep in range(3):
                config = ExperimentConfig(
                    experiment_id=f"direct_compare_{agent_type.value}_{rep+1:02d}",
                    test_type=TestType.LLM_COMPARISON,
                    agent_type=agent_type,
                    consensus_protocol=ConsensusProtocol.BYZANTINE_FAULT_TOLERANT,
                    agent_count=5,
                    job_count=10,
                    fault_type=FaultType.NONE,
                    fault_intensity=0.0,
                    duration_seconds=300,
                    specialization_enabled=True,
                    llm_temperature=0.1,  # Use demo temperature for consistency
                    repetition=rep
                )
                self.experiments.append(config)
    
    async def run_campaign(self):
        """Execute all experiments in the campaign"""
        self.logger.info(f"Starting campaign '{self.campaign_name}' with {len(self.experiments)} experiments")
        
        start_time = time.time()
        
        for i, exp_config in enumerate(self.experiments, 1):
            self.logger.info(f"Running experiment {i}/{len(self.experiments)}: {exp_config.experiment_id}")
            
            try:
                runner = ExperimentRunner(exp_config)
                runner.setup_experiment()
                result = await runner.run_experiment()
                self.results.append(result)
                
                # Save individual result
                result_file = self.output_dir / f"{exp_config.experiment_id}_result.json"
                with open(result_file, 'w') as f:
                    json.dump(asdict(result), f, indent=2, default=str)
                
            except Exception as e:
                self.logger.error(f"Experiment {exp_config.experiment_id} failed: {e}")
                # Create error result
                error_result = ExperimentResults(
                    experiment_id=exp_config.experiment_id,
                    config=exp_config,
                    success_rate=0.0,
                    consensus_time_avg=0.0,
                    consensus_time_std=0.0,
                    throughput=0.0,
                    fault_detection_rate=0.0,
                    recovery_time_avg=0.0,
                    llm_response_time_avg=0.0,
                    llm_success_rate=0.0,
                    resource_utilization=0.0,
                    agent_performance={},
                    timeline=[],
                    error_log=[str(e)],
                    start_time=datetime.now(),
                    end_time=datetime.now()
                )
                self.results.append(error_result)
        
        total_time = time.time() - start_time
        self.logger.info(f"Campaign completed in {total_time:.2f} seconds")
        
        # Generate final reports
        await self._generate_reports()
    
    async def _generate_reports(self):
        """Generate comprehensive campaign reports"""
        self.logger.info("Generating campaign reports")
        
        # Save aggregated results
        aggregated_file = self.output_dir / f"{self.campaign_name}_aggregated_results.json"
        with open(aggregated_file, 'w') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2, default=str)
        
        # Generate summary report
        summary = self._generate_summary()
        summary_file = self.output_dir / f"{self.campaign_name}_summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        # Generate analytics if available
        if ANALYTICS_AVAILABLE:
            self._generate_analytics()
        
        self.logger.info(f"Reports generated in {self.output_dir}")
    
    def _generate_summary(self) -> str:
        """Generate markdown summary report"""
        total_experiments = len(self.results)
        successful_experiments = len([r for r in self.results if not r.error_log])
        
        avg_success_rate = np.mean([r.success_rate for r in self.results]) if self.results and ANALYTICS_AVAILABLE else 0.0
        avg_consensus_time = np.mean([r.consensus_time_avg for r in self.results]) if self.results and ANALYTICS_AVAILABLE else 0.0
        avg_throughput = np.mean([r.throughput for r in self.results]) if self.results and ANALYTICS_AVAILABLE else 0.0
        
        summary = f"""# Experimental Campaign Report: {self.campaign_name}

## Overview
- **Total Experiments**: {total_experiments}
- **Successful Experiments**: {successful_experiments}
- **Success Rate**: {(successful_experiments/total_experiments)*100:.1f}%
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Metrics
- **Average Success Rate**: {avg_success_rate:.1%}
- **Average Consensus Time**: {avg_consensus_time:.2f} seconds
- **Average Throughput**: {avg_throughput:.2f} jobs/second

## Test Categories
"""
        
        # Group results by test type
        by_test_type = {}
        for result in self.results:
            test_type = result.config.test_type.value
            if test_type not in by_test_type:
                by_test_type[test_type] = []
            by_test_type[test_type].append(result)
        
        for test_type, results in by_test_type.items():
            if ANALYTICS_AVAILABLE:
                avg_success = np.mean([r.success_rate for r in results])
                avg_time = np.mean([r.consensus_time_avg for r in results])
            else:
                avg_success = sum(r.success_rate for r in results) / len(results)
                avg_time = sum(r.consensus_time_avg for r in results) / len(results)
            
            summary += f"""
### {test_type.replace('_', ' ').title()}
- **Experiments**: {len(results)}
- **Average Success Rate**: {avg_success:.1%}
- **Average Consensus Time**: {avg_time:.2f}s
"""
        
        summary += f"""
## Detailed Results
Results saved to: `{self.campaign_name}_aggregated_results.json`

## Files Generated
- `{self.campaign_name}_summary.md` - This summary report
- `{self.campaign_name}_aggregated_results.json` - Detailed results data
- Individual experiment results: `[experiment_id]_result.json`
"""
        
        return summary
    
    def _generate_analytics(self):
        """Generate analytics and visualizations if pandas/matplotlib available"""
        if not ANALYTICS_AVAILABLE or not VISUALIZATION_AVAILABLE:
            return
        
        # Create DataFrame from results
        data = []
        for result in self.results:
            data.append({
                'experiment_id': result.experiment_id,
                'test_type': result.config.test_type.value,
                'agent_type': result.config.agent_type.value,
                'consensus_protocol': result.config.consensus_protocol.value,
                'agent_count': result.config.agent_count,
                'fault_type': result.config.fault_type.value,
                'fault_intensity': result.config.fault_intensity,
                'success_rate': result.success_rate,
                'consensus_time_avg': result.consensus_time_avg,
                'throughput': result.throughput,
                'llm_success_rate': result.llm_success_rate
            })
        
        df = pd.DataFrame(data)
        
        # Generate plots
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
        
        # Success rate by agent type
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='agent_type', y='success_rate')
        plt.title('Success Rate by Agent Type')
        plt.ylabel('Success Rate')
        plt.savefig(self.output_dir / 'success_rate_by_agent_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Consensus time by protocol
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='consensus_protocol', y='consensus_time_avg')
        plt.title('Consensus Time by Protocol')
        plt.ylabel('Average Consensus Time (seconds)')
        plt.xticks(rotation=45)
        plt.savefig(self.output_dir / 'consensus_time_by_protocol.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Scalability analysis
        scale_data = df[df['test_type'] == 'scalability']
        if not scale_data.empty:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=scale_data, x='agent_count', y='throughput', s=100)
            plt.title('Scalability: Throughput vs Agent Count')
            plt.xlabel('Number of Agents')
            plt.ylabel('Throughput (jobs/second)')
            plt.savefig(self.output_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()


async def run_llm_vs_heuristic_comparison():
    """Run focused LLM vs Heuristic comparison"""
    print("🔬 LLM vs Heuristic Agent Comparison Test")
    print("=" * 60)
    
    # Check environment
    api_key = os.getenv('SAMBASTUDIO_API_KEY')
    api_url = os.getenv('SAMBASTUDIO_URL')
    
    if not api_key or not api_url:
        print("⚠️ Warning: SambaNova environment variables not set!")
        print("   SAMBASTUDIO_API_KEY:", "✅ SET" if api_key else "❌ MISSING")
        print("   SAMBASTUDIO_URL:", "✅ SET" if api_url else "❌ MISSING")
        print("\n💡 To load environment variables, run:")
        print("   source ~/.bashrc && python evaluation/llm_experimental_framework.py --llm-vs-heuristic")
        print()
    
    # Create focused campaign
    campaign = ExperimentalCampaign("llm_vs_heuristic_comparison", "llm_comparison_results")
    
    # Add direct comparison tests
    campaign.add_direct_llm_vs_heuristic_test()
    
    print(f"📋 Campaign configured with {len(campaign.experiments)} experiments")
    print("🚀 Starting LLM vs Heuristic comparison...")
    
    # Run the campaign
    await campaign.run_campaign()
    
    # Generate comparison analysis
    await _generate_comparison_analysis(campaign)
    
    print("🎉 LLM vs Heuristic comparison completed!")
    print(f"📊 Results available in: {campaign.output_dir}")

async def _generate_comparison_analysis(campaign: ExperimentalCampaign):
    """Generate specific analysis for LLM vs Heuristic comparison"""
    heuristic_results = [r for r in campaign.results if r.config.agent_type == AgentType.HEURISTIC]
    llm_results = [r for r in campaign.results if r.config.agent_type == AgentType.LLM]
    
    if not heuristic_results or not llm_results:
        campaign.logger.warning("Insufficient results for comparison analysis")
        return
    
    # Calculate averages
    if ANALYTICS_AVAILABLE:
        h_success = np.mean([r.success_rate for r in heuristic_results])
        l_success = np.mean([r.success_rate for r in llm_results])
        h_time = np.mean([r.consensus_time_avg for r in heuristic_results])
        l_time = np.mean([r.consensus_time_avg for r in llm_results])
        h_throughput = np.mean([r.throughput for r in heuristic_results])
        l_throughput = np.mean([r.throughput for r in llm_results])
        l_llm_success = np.mean([r.llm_success_rate for r in llm_results])
    else:
        h_success = sum(r.success_rate for r in heuristic_results) / len(heuristic_results)
        l_success = sum(r.success_rate for r in llm_results) / len(llm_results)
        h_time = sum(r.consensus_time_avg for r in heuristic_results) / len(heuristic_results)
        l_time = sum(r.consensus_time_avg for r in llm_results) / len(llm_results)
        h_throughput = sum(r.throughput for r in heuristic_results) / len(heuristic_results)
        l_throughput = sum(r.throughput for r in llm_results) / len(llm_results)
        l_llm_success = sum(r.llm_success_rate for r in llm_results) / len(llm_results)
    
    # Generate detailed comparison report
    comparison_report = f"""# LLM vs Heuristic Agent Comparison Analysis

## Test Configuration
- **Heuristic Tests**: {len(heuristic_results)} experiments
- **LLM Tests**: {len(llm_results)} experiments
- **Agent Count**: {heuristic_results[0].config.agent_count if heuristic_results else 'N/A'}
- **Jobs per Test**: {heuristic_results[0].config.job_count if heuristic_results else 'N/A'}
- **Consensus Protocol**: {heuristic_results[0].config.consensus_protocol.value if heuristic_results else 'N/A'}

## Performance Comparison

| Metric | Heuristic | LLM | Difference | Winner |
|--------|-----------|-----|------------|--------|
| Success Rate | {h_success:.1%} | {l_success:.1%} | {l_success - h_success:+.1%} | {'🤖 LLM' if l_success > h_success else '⚙️ Heuristic' if h_success > l_success else '🤝 Tie'} |
| Avg Consensus Time | {h_time:.3f}s | {l_time:.3f}s | {l_time - h_time:+.3f}s | {'⚙️ Heuristic' if h_time < l_time else '🤖 LLM' if l_time < h_time else '🤝 Tie'} |
| Throughput | {h_throughput:.2f} jobs/s | {l_throughput:.2f} jobs/s | {l_throughput - h_throughput:+.2f} jobs/s | {'🤖 LLM' if l_throughput > h_throughput else '⚙️ Heuristic' if h_throughput > l_throughput else '🤝 Tie'} |
| LLM API Success | N/A | {l_llm_success:.1%} | - | - |

## Key Findings

"""
    
    if l_success > h_success:
        comparison_report += f"- ✅ **LLM Advantage**: LLM agents achieved {(l_success - h_success)*100:.1f} percentage points higher success rate\n"
    elif h_success > l_success:
        comparison_report += f"- ⚙️ **Heuristic Advantage**: Heuristic agents achieved {(h_success - l_success)*100:.1f} percentage points higher success rate\n"
    else:
        comparison_report += f"- 🤝 **Equal Performance**: Both agent types achieved equal success rates\n"
    
    if l_time < h_time:
        comparison_report += f"- ⚡ **Speed Advantage**: LLM agents were {(h_time - l_time)*1000:.0f}ms faster per consensus\n"
    elif h_time < l_time:
        comparison_report += f"- 🐌 **Speed Disadvantage**: LLM agents took {(l_time - h_time)*1000:.0f}ms longer per consensus\n"
    
    comparison_report += f"- 🤖 **LLM Reliability**: {l_llm_success:.1%} of LLM API calls succeeded\n"
    
    if l_llm_success < 0.8:
        comparison_report += f"- ⚠️ **API Issues**: Low LLM success rate suggests API connectivity or configuration issues\n"
    
    comparison_report += f"""
## Recommendations

{'- 🎯 **Use LLM Agents**: Higher success rate justifies the additional latency' if l_success > h_success else '- ⚙️ **Use Heuristic Agents**: Better or equal performance with lower latency'}
- 🔧 **Optimization**: {'Improve LLM API reliability' if l_llm_success < 0.9 else 'Consider hybrid approach for optimal performance'}
- 📈 **Scaling**: {'LLM agents show promise for complex scenarios' if l_success > h_success else 'Heuristic agents provide consistent baseline performance'}

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Save comparison report
    comparison_file = campaign.output_dir / "llm_vs_heuristic_analysis.md"
    with open(comparison_file, 'w') as f:
        f.write(comparison_report)
    
    campaign.logger.info(f"Detailed comparison analysis saved to {comparison_file}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("📈 COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'Heuristic':<12} {'LLM':<12} {'Winner':<15}")
    print("-" * 59)
    success_winner = '🤖 LLM' if l_success > h_success else '⚙️ Heuristic' if h_success > l_success else '🤝 Tie'
    time_winner = '⚙️ Heuristic' if h_time < l_time else '🤖 LLM' if l_time < h_time else '🤝 Tie'
    throughput_winner = '🤖 LLM' if l_throughput > h_throughput else '⚙️ Heuristic' if h_throughput > l_throughput else '🤝 Tie'
    
    h_success_str = f"{h_success:.1%}"
    l_success_str = f"{l_success:.1%}"
    h_time_str = f"{h_time:.3f}s"
    l_time_str = f"{l_time:.3f}s"
    h_throughput_str = f"{h_throughput:.1f}/s"
    l_throughput_str = f"{l_throughput:.1f}/s"
    l_llm_success_str = f"{l_llm_success:.1%}"
    
    print(f"{'Success Rate':<20} {h_success_str:<12} {l_success_str:<12} {success_winner:<15}")
    print(f"{'Consensus Time':<20} {h_time_str:<12} {l_time_str:<12} {time_winner:<15}")
    print(f"{'Throughput':<20} {h_throughput_str:<12} {l_throughput_str:<12} {throughput_winner:<15}")
    print(f"{'LLM API Success':<20} {'N/A':<12} {l_llm_success_str:<12} {'':<15}")

async def run_full_experimental_campaign():
    """Run the complete 5-phase experimental campaign from EXPERIMENTS.md"""
    print("🧪 LLM-Enhanced Distributed Consensus Experimental Campaign")
    print("=" * 70)
    print("Based on EXPERIMENTS.md - Comprehensive 5-Phase Design")
    print()
    
    # Display research questions
    print("🔬 Research Questions:")
    for rq_id, rq_data in RESEARCH_QUESTIONS.items():
        print(f"   {rq_id}: {rq_data['question']}")
        print(f"       Expected: {rq_data['expected_outcome']}")
    print()
    
    # Create experimental campaign
    campaign = ExperimentalCampaign("llm_consensus_evaluation_2025", "experimental_campaign_results")
    
    # Add all 5 phases from EXPERIMENTS.md
    campaign.add_phase_1_baseline_establishment()
    campaign.add_phase_2_llm_evaluation() 
    campaign.add_phase_3_specialization_analysis()
    campaign.add_phase_4_scalability_stress()
    campaign.add_phase_5_validation_robustness()
    
    print(f"📋 Campaign configured with {len(campaign.experiments)} experiments")
    print("📊 Phase breakdown:")
    
    # Count experiments by phase
    phase_counts = {}
    for exp in campaign.experiments:
        phase = exp.test_type.value
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    for phase, count in phase_counts.items():
        print(f"   {phase}: {count} experiments")
    
    # Estimated duration from EXPERIMENTS.md
    print("\n⏱️ Estimated Campaign Duration: 11 weeks")
    print("💰 Total Computational Budget: 2,500 hours")
    print("📈 Expected Outcomes: 2,496+ experimental runs")
    print()
    
    # Run the campaign
    await campaign.run_campaign()
    
    # Generate research question analysis
    await _analyze_research_questions(campaign)
    
    print("🎉 Experimental campaign completed successfully!")
    print(f"📊 Results available in: {campaign.output_dir}")

async def _analyze_research_questions(campaign: ExperimentalCampaign):
    """Analyze results in context of research questions from EXPERIMENTS.md"""
    campaign.logger.info("Analyzing results for research questions")
    
    rq_analysis = {}
    
    for rq_id, rq_data in RESEARCH_QUESTIONS.items():
        # Filter results for this research question
        relevant_results = [r for r in campaign.results if r.config.test_type == rq_data['test_type']]
        
        if not relevant_results:
            rq_analysis[rq_id] = {"status": "no_data", "message": "No relevant experiments completed"}
            continue
        
        # Analyze based on research question
        if rq_id == "RQ1":  # LLM vs. Heuristic Performance
            rq_analysis[rq_id] = _analyze_rq1_llm_vs_heuristic(relevant_results)
        elif rq_id == "RQ2":  # Protocol Effectiveness
            rq_analysis[rq_id] = _analyze_rq2_protocol_effectiveness(relevant_results)
        elif rq_id == "RQ3":  # Specialization Impact
            rq_analysis[rq_id] = _analyze_rq3_specialization_impact(relevant_results)
        elif rq_id == "RQ4":  # Scalability & Resilience
            rq_analysis[rq_id] = _analyze_rq4_scalability_resilience(relevant_results)
    
    # Generate research question report
    rq_report = f"""# Research Question Analysis

## Campaign Overview
- **Campaign ID**: {campaign.campaign_name}
- **Total Experiments**: {len(campaign.results)}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    
    for rq_id, analysis in rq_analysis.items():
        rq_data = RESEARCH_QUESTIONS[rq_id]
        rq_report += f"""## {rq_id}: {rq_data['question']}

**Focus Area**: {rq_data['focus_area']}  
**Expected Outcome**: {rq_data['expected_outcome']}  
**Status**: {analysis.get('status', 'analyzed')}

"""
        if 'findings' in analysis:
            rq_report += f"**Key Findings**:\n"
            for finding in analysis['findings']:
                rq_report += f"- {finding}\n"
            rq_report += "\n"
    
    # Save research question analysis
    rq_file = campaign.output_dir / "research_question_analysis.md"
    with open(rq_file, 'w') as f:
        f.write(rq_report)
    
    campaign.logger.info(f"Research question analysis saved to {rq_file}")

def _analyze_rq1_llm_vs_heuristic(results):
    """Analyze RQ1: LLM vs. Heuristic Performance (15-25% improvement expected)"""
    llm_results = [r for r in results if r.config.agent_type == AgentType.LLM]
    heuristic_results = [r for r in results if r.config.agent_type == AgentType.HEURISTIC]
    
    if not llm_results or not heuristic_results:
        return {"status": "insufficient_data", "findings": ["Need both LLM and heuristic results for comparison"]}
    
    if ANALYTICS_AVAILABLE:
        llm_success = np.mean([r.success_rate for r in llm_results])
        heuristic_success = np.mean([r.success_rate for r in heuristic_results])
    else:
        llm_success = sum(r.success_rate for r in llm_results) / len(llm_results)
        heuristic_success = sum(r.success_rate for r in heuristic_results) / len(heuristic_results)
    
    improvement = ((llm_success - heuristic_success) / heuristic_success) * 100
    target_met = 15 <= improvement <= 25
    
    return {
        "status": "supported" if target_met else "partially_supported",
        "findings": [
            f"LLM agents achieved {improvement:.1f}% improvement over heuristic agents",
            f"Target range: 15-25%, Actual: {improvement:.1f}%",
            f"LLM success rate: {llm_success:.1%}, Heuristic: {heuristic_success:.1%}",
            "✅ Hypothesis supported" if target_met else "⚠️ Improvement outside expected range"
        ]
    }

def _analyze_rq2_protocol_effectiveness(results):
    """Analyze RQ2: Protocol Effectiveness (BFT best for adversarial, Raft best for crashes)"""
    protocol_performance = {}
    
    for result in results:
        protocol = result.config.consensus_protocol.value
        if protocol not in protocol_performance:
            protocol_performance[protocol] = []
        protocol_performance[protocol].append(result.success_rate)
    
    findings = []
    for protocol, rates in protocol_performance.items():
        if ANALYTICS_AVAILABLE:
            avg_rate = np.mean(rates)
        else:
            avg_rate = sum(rates) / len(rates)
        findings.append(f"{protocol}: {avg_rate:.1%} average success rate")
    
    return {
        "status": "analyzed",
        "findings": findings + ["Protocol ranking analysis completed"]
    }

def _analyze_rq3_specialization_impact(results):
    """Analyze RQ3: Specialization Impact (30-40% accuracy improvement expected)"""
    specialized_results = [r for r in results if r.config.specialization_enabled]
    non_specialized_results = [r for r in results if not r.config.specialization_enabled]
    
    if not specialized_results or not non_specialized_results:
        return {"status": "insufficient_data", "findings": ["Need both specialized and non-specialized results"]}
    
    if ANALYTICS_AVAILABLE:
        spec_success = np.mean([r.success_rate for r in specialized_results])
        non_spec_success = np.mean([r.success_rate for r in non_specialized_results])
    else:
        spec_success = sum(r.success_rate for r in specialized_results) / len(specialized_results)
        non_spec_success = sum(r.success_rate for r in non_specialized_results) / len(non_specialized_results)
    
    improvement = ((spec_success - non_spec_success) / non_spec_success) * 100
    target_met = 30 <= improvement <= 40
    
    return {
        "status": "supported" if target_met else "partially_supported",
        "findings": [
            f"Specialization achieved {improvement:.1f}% improvement",
            f"Target range: 30-40%, Actual: {improvement:.1f}%",
            f"Specialized: {spec_success:.1%}, Non-specialized: {non_spec_success:.1%}",
            "✅ Hypothesis supported" if target_met else "⚠️ Improvement outside expected range"
        ]
    }

def _analyze_rq4_scalability_resilience(results):
    """Analyze RQ4: Scalability & Resilience (>80% at 50+ agents, >70% at 40% faults)"""
    # Analyze high agent count performance
    high_agent_results = [r for r in results if r.config.agent_count >= 50]
    high_fault_results = [r for r in results if r.config.fault_intensity >= 0.4]
    
    findings = []
    
    if high_agent_results:
        if ANALYTICS_AVAILABLE:
            high_agent_perf = np.mean([r.success_rate for r in high_agent_results])
        else:
            high_agent_perf = sum(r.success_rate for r in high_agent_results) / len(high_agent_results)
        
        agent_target_met = high_agent_perf >= 0.8
        findings.append(f"50+ agents performance: {high_agent_perf:.1%} (target: >80%)")
        findings.append("✅ Agent scalability target met" if agent_target_met else "❌ Agent scalability target not met")
    
    if high_fault_results:
        if ANALYTICS_AVAILABLE:
            high_fault_perf = np.mean([r.success_rate for r in high_fault_results])
        else:
            high_fault_perf = sum(r.success_rate for r in high_fault_results) / len(high_fault_results)
        
        fault_target_met = high_fault_perf >= 0.7
        findings.append(f"40%+ fault performance: {high_fault_perf:.1%} (target: >70%)")
        findings.append("✅ Fault resilience target met" if fault_target_met else "❌ Fault resilience target not met")
    
    return {
        "status": "analyzed",
        "findings": findings if findings else ["Insufficient high-stress test data"]
    }

async def main():
    """Main function demonstrating the experimental framework"""
    print("🧪 LLM-Enabled Multiagent Fault Tolerance Experimental Framework")
    print("=" * 70)
    
    # Create experimental campaign
    campaign = ExperimentalCampaign("llm_multiagent_evaluation", "experiment_results")
    
    # Add different types of tests
    campaign.add_phase_1_baseline_establishment()
    campaign.add_phase_2_llm_evaluation()
    campaign.add_phase_3_specialization_analysis()
    campaign.add_llm_comparison_tests()
    
    print(f"📋 Campaign configured with {len(campaign.experiments)} experiments")
    
    # Run the campaign
    await campaign.run_campaign()
    
    print("🎉 Experimental campaign completed successfully!")
    print(f"📊 Results available in: {campaign.output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Multiagent Experimental Framework")
    parser.add_argument("--campaign-name", default="llm_multiagent_test", help="Name of the experimental campaign")
    parser.add_argument("--output-dir", default="experiment_results", help="Output directory for results")
    parser.add_argument("--quick", action="store_true", help="Run a quick test with fewer experiments")
    parser.add_argument("--llm-vs-heuristic", action="store_true", help="Run focused LLM vs Heuristic comparison")
    parser.add_argument("--full-campaign", action="store_true", help="Run complete 5-phase campaign from EXPERIMENTS.md")
    
    args = parser.parse_args()
    
    if args.full_campaign:
        print("🚀 Running complete 5-phase experimental campaign")
        asyncio.run(run_full_experimental_campaign())
    elif args.llm_vs_heuristic:
        print("🔬 Running LLM vs Heuristic comparison")
        asyncio.run(run_llm_vs_heuristic_comparison())
    elif args.quick:
        print("🚀 Running quick test mode")
        # Run a single experiment for testing
        config = ExperimentConfig(
            experiment_id="quick_test",
            test_type=TestType.BASELINE_PERFORMANCE,
            agent_type=AgentType.LLM,
            consensus_protocol=ConsensusProtocol.BYZANTINE_FAULT_TOLERANT,
            agent_count=3,
            job_count=5,
            fault_type=FaultType.NONE,
            fault_intensity=0.0,
            duration_seconds=60,
            specialization_enabled=True,
            llm_temperature=0.0,
            repetition=1
        )
        
        async def quick_test():
            runner = ExperimentRunner(config)
            runner.setup_experiment()
            result = await runner.run_experiment()
            
            print("\n🎯 Quick Test Results:")
            print(f"   Success Rate: {result.success_rate:.1%}")
            print(f"   Consensus Time: {result.consensus_time_avg:.2f}s")
            print(f"   Throughput: {result.throughput:.2f} jobs/sec")
        
        asyncio.run(quick_test())
    else:
        asyncio.run(main())
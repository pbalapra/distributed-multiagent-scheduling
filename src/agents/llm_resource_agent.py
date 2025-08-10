#!/usr/bin/env python3
"""
LLM-Enhanced Resource Agent
===========================

Extends the base resource agent with Large Language Model capabilities
for intelligent decision-making in job scoring, fault recovery, and negotiation.
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from .resource_agent import ResourceAgent
from ..llm.llm_interface import LLMRequest, llm_manager, LLMProvider
from ..llm.context_manager import ContextBuilder
from ..jobs.job import Job, JobStatus, JobPriority
from ..resources.resource import Resource
from ..communication.protocol import Message, MessageType, MessagePriority


class LLMResourceAgent(ResourceAgent):
    """Resource agent enhanced with LLM decision-making capabilities"""
    
    def __init__(self, agent_id: str, message_bus, resource: Resource,
                 failure_rate: float = 0.05, 
                 llm_enabled: bool = True,
                 llm_provider: Optional[LLMProvider] = None,
                 context_window_size: int = 1000):
        """
        Initialize LLM-enhanced resource agent
        
        Args:
            agent_id: Unique identifier for this agent
            message_bus: Communication bus for message passing
            resource: Resource this agent manages
            failure_rate: Probability of random failures
            llm_enabled: Whether to use LLM for decision making
            llm_provider: Specific LLM provider to use
            context_window_size: Size of context history to maintain
        """
        super().__init__(agent_id, message_bus, resource)
        
        # Initialize pending_jobs if not present
        if not hasattr(self, 'pending_jobs'):
            self.pending_jobs = []
        
        self.llm_enabled = llm_enabled
        self.llm_provider = llm_provider
        self.context_builder = ContextBuilder(max_history_size=context_window_size)
        
        # Enhanced state tracking for LLM context
        self.decision_history: List[Dict] = []
        self.performance_metrics = {
            "successful_jobs": 0,
            "failed_jobs": 0,
            "average_completion_time": 0.0,
            "utilization_history": [],
            "decision_accuracy": 0.0
        }
        
        # LLM-specific configuration
        self.llm_config = {
            "job_scoring": {
                "temperature": 0.1,
                "max_tokens": 500,
                "timeout": 10.0
            },
            "fault_recovery": {
                "temperature": 0.2,
                "max_tokens": 800,
                "timeout": 15.0
            },
            "negotiation": {
                "temperature": 0.3,
                "max_tokens": 600,
                "timeout": 12.0
            }
        }
        
        print(f"🧠 LLM-Enhanced Resource Agent {agent_id} initialized "
              f"(LLM: {'enabled' if llm_enabled else 'disabled'})")
    
    def _calculate_job_score(self, job_data: Dict) -> float:
        """
        Enhanced job scoring using LLM reasoning
        Falls back to original heuristic if LLM is disabled or fails
        """
        if not self.llm_enabled:
            return super()._calculate_job_score(job_data)
        
        try:
            # Build rich context for LLM decision making
            system_state = {
                "queue_length": len(self.pending_jobs),
                "average_wait_time": self._calculate_average_wait_time(),
                "system_load": self.resource.utilization_score,
                "available_agents": 1 if self.resource.utilization_score < 0.9 else 0,
                "busy_agents": 1 if self.resource.utilization_score >= 0.9 else 0
            }
            
            context = self.context_builder.build_job_scoring_context(
                job_data, self.resource, system_state
            )
            
            # Create LLM request
            prompt = self._build_job_scoring_prompt(context)
            request = LLMRequest(
                prompt=prompt,
                context=context,
                task_type="job_scoring",
                **self.llm_config["job_scoring"]
            )
            
            # Get LLM response
            response = llm_manager.generate_sync(request, self.llm_provider)
            
            # Parse response and extract score
            result = json.loads(response.content)
            score = result.get("score", 0.0)
            
            # Record decision for learning
            self.context_builder.record_decision(
                "job_scoring", context, result, None
            )
            
            # Log reasoning if available
            if response.reasoning:
                print(f"🧠 Agent {self.agent_id} job scoring reasoning: {response.reasoning[:100]}...")
            
            return float(score)
            
        except Exception as e:
            print(f"⚠️ LLM job scoring failed for agent {self.agent_id}: {e}")
            print("🔄 Falling back to heuristic scoring")
            return super()._calculate_job_score(job_data)
    
    def _handle_job_failure(self, job_id: str, failure_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced fault recovery using LLM reasoning
        """
        if not self.llm_enabled:
            return self._basic_failure_recovery(job_id, failure_info)
        
        try:
            # Get job information
            job = None
            for pending_job in self.pending_jobs:
                if pending_job.job_id == job_id:
                    job = pending_job
                    break
            
            if not job:
                print(f"⚠️ Job {job_id} not found for failure recovery")
                return {"action": "ignore", "reason": "job_not_found"}
            
            # Get available resources (mock - in real system, query scheduler)
            available_resources = self._get_available_alternative_resources()
            
            # Build context for LLM
            system_state = {
                "recent_failure_rate": 0.05,  # Mock value
                "system_stress": self.resource.utilization_score,
                "recovery_resources_available": len(available_resources)
            }
            
            context = self.context_builder.build_fault_recovery_context(
                failure_info, job, available_resources, system_state
            )
            
            # Create LLM request
            prompt = self._build_fault_recovery_prompt(context)
            request = LLMRequest(
                prompt=prompt,
                context=context,
                task_type="fault_recovery",
                **self.llm_config["fault_recovery"]
            )
            
            # Get LLM response
            response = llm_manager.generate_sync(request, self.llm_provider)
            result = json.loads(response.content)
            
            # Record decision
            self.context_builder.record_decision(
                "fault_recovery", context, result, None
            )
            
            print(f"🧠 Agent {self.agent_id} fault recovery decision: {result.get('strategy', 'unknown')}")
            if response.reasoning:
                print(f"   Reasoning: {response.reasoning[:150]}...")
            
            return result
            
        except Exception as e:
            print(f"⚠️ LLM fault recovery failed for agent {self.agent_id}: {e}")
            print("🔄 Falling back to basic recovery")
            return self._basic_failure_recovery(job_id, failure_info)
    
    def _evaluate_proposal(self, proposal: Dict) -> Dict[str, Any]:
        """
        Enhanced proposal evaluation using LLM strategic reasoning
        """
        if not self.llm_enabled:
            basic_result = super()._evaluate_proposal(proposal)
            return {"accept": basic_result, "reasoning": "heuristic_decision"}
        
        try:
            # Build agent state
            agent_state = {
                "utilization": self.resource.utilization_score,
                "current_utilization": self.resource.utilization_score,
                "available_capacity": {
                    "cpu": self.resource.capacity.total_cpu_cores * (1 - self.resource.utilization_score),
                    "memory": self.resource.capacity.total_memory_gb * (1 - self.resource.utilization_score)
                },
                "scheduled_jobs": len(self.pending_jobs),
                "revenue_target": 1000.0,  # Mock value
                "reputation_score": 0.85,  # Mock value
                "strategic_goals": ["maximize_utilization", "maintain_quality"]
            }
            
            # Mock market conditions
            market_conditions = {
                "demand_level": "medium",
                "supply_availability": "medium",
                "average_prices": {"cpu_hour": 0.1, "memory_gb_hour": 0.05},
                "competitor_activity": {"active_agents": 5, "average_utilization": 0.7},
                "is_peak_hour": datetime.now().hour in [9, 10, 11, 14, 15, 16]
            }
            
            # Build context
            context = self.context_builder.build_negotiation_context(
                proposal, agent_state, market_conditions
            )
            
            # Create LLM request
            prompt = self._build_negotiation_prompt(context)
            request = LLMRequest(
                prompt=prompt,
                context=context,
                task_type="negotiation",
                **self.llm_config["negotiation"]
            )
            
            # Get LLM response
            response = llm_manager.generate_sync(request, self.llm_provider)
            result = json.loads(response.content)
            
            # Record decision
            self.context_builder.record_decision(
                "negotiation", context, result, None
            )
            
            print(f"🧠 Agent {self.agent_id} negotiation decision: "
                  f"{'ACCEPT' if result.get('accept') else 'REJECT'}")
            if response.reasoning:
                print(f"   Reasoning: {response.reasoning[:150]}...")
            
            return result
            
        except Exception as e:
            print(f"⚠️ LLM negotiation failed for agent {self.agent_id}: {e}")
            print("🔄 Falling back to heuristic evaluation")
            basic_result = super()._evaluate_proposal(proposal)
            return {"accept": basic_result, "reasoning": "fallback_heuristic"}
    
    def _build_job_scoring_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for job scoring task"""
        job = context["job"]
        resource = context["resource"]
        system = context["system_state"]
        
        prompt = f"""
You are an intelligent resource allocation agent managing a high-performance computing resource.

TASK: Score how well this job matches your resource capabilities (0.0 to 1.0 scale).

JOB DETAILS:
- Job ID: {job['id']}
- Priority: {job['priority']} (1=low, 4=critical)
- CPU Required: {job['requirements'].get('cpu_cores', 0)} cores
- Memory Required: {job['requirements'].get('memory_gb', 0)} GB
- Estimated Runtime: {job['requirements'].get('estimated_runtime_minutes', 0)} minutes

RESOURCE STATUS:
- Available CPU: {resource['available_cpu']:.1f} cores
- Available Memory: {resource['available_memory']:.1f} GB
- Current Utilization: {resource['current_utilization']:.2f}
- Location: {resource['location']}
- Cost per Hour: ${resource['cost_per_hour']:.2f}

SYSTEM STATE:
- Queue Length: {system['queue_length']} jobs
- System Load: {system['system_load']:.2f}
- Available Agents: {system['available_agents']}

OBJECTIVES:
- Maximize resource efficiency
- Respect job priorities
- Balance workload
- Minimize wait times

Return a JSON response with:
{{"score": <float 0.0-1.0>, "factors": {{}}, "recommendation": "<accept|consider_alternatives>"}}

Focus on resource compatibility, current load, and strategic fit.
"""
        return prompt
    
    def _build_fault_recovery_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for fault recovery task"""
        failure = context["failure"]
        job = context["job"]
        resources = context["available_resources"]
        
        prompt = f"""
You are a fault-tolerant job scheduler handling a job execution failure.

TASK: Determine the best recovery strategy for a failed job.

FAILURE DETAILS:
- Type: {failure['type']}
- Failed Resource: {failure['failed_resource']}
- Retry Count: {failure['retry_count']}
- Error: {failure.get('error_message', 'Unknown error')}

JOB DETAILS:
- Job ID: {job['id']}
- Priority: {job['priority']} (1=low, 4=critical)
- Criticality: {job['criticality']}
- Retry Tolerance: {job.get('retry_tolerance', 3)}

AVAILABLE ALTERNATIVES:
{json.dumps(resources[:3], indent=2)}

RECOVERY OPTIONS:
- immediate_retry: Retry immediately on same/different resource
- delayed_retry: Wait and retry with exponential backoff
- alternative_resource: Switch to different resource
- job_modification: Modify job requirements
- escalate: Require human intervention

Return JSON response with:
{{"strategy": "<recovery_option>", "action": "<specific_action>", "alternative_resource": "<resource_id>", "delay_seconds": <int>, "escalate": <bool>}}

Consider failure patterns, job criticality, and resource availability.
"""
        return prompt
    
    def _build_negotiation_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for negotiation task"""
        proposal = context["proposal"]
        agent = context["agent_state"]
        market = context["market_conditions"]
        
        prompt = f"""
You are a strategic resource agent negotiating job execution contracts.

TASK: Decide whether to accept a job proposal and potentially make a counter-offer.

PROPOSAL:
- Job ID: {proposal['job_id']}
- Priority: {proposal['priority']}
- Duration: {proposal['estimated_duration']} minutes
- Offered Price: ${proposal['offered_price']:.2f}
- From: {proposal['sender']}
- Round: {proposal['negotiation_round']}

YOUR STATUS:
- Current Utilization: {agent['current_utilization']:.2f}
- Available CPU: {agent['available_capacity'].get('cpu', 0)} cores
- Available Memory: {agent['available_capacity'].get('memory', 0)} GB
- Revenue Target: ${agent['revenue_target']:.2f}
- Reputation Score: {agent['reputation_score']:.2f}

MARKET CONDITIONS:
- Demand Level: {market['demand_level']}
- Supply Available: {market['supply_availability']}
- Peak Hour: {market['peak_hours']}
- Average CPU Price: ${market['average_prices'].get('cpu_hour', 0):.3f}/hour

STRATEGIC GOALS:
- Maximize revenue while maintaining service quality
- Build long-term partnerships
- Optimize resource utilization
- Maintain competitive positioning

Return JSON response with:
{{"accept": <bool>, "counter_offer": {{}}, "reasoning_factors": {{}}}}

Consider current capacity, market conditions, and strategic value.
"""
        return prompt
    
    def _basic_failure_recovery(self, job_id: str, failure_info: Dict[str, Any]) -> Dict[str, Any]:
        """Basic failure recovery fallback"""
        retry_count = failure_info.get("retry_count", 0)
        
        if retry_count < 3:
            return {
                "strategy": "retry_with_alternative",
                "action": "immediate_retry",
                "delay_seconds": 0,
                "escalate": False
            }
        else:
            return {
                "strategy": "escalate",
                "action": "human_intervention",
                "delay_seconds": 0,
                "escalate": True
            }
    
    def _get_available_alternative_resources(self) -> List[Dict]:
        """Get list of available alternative resources (mock implementation)"""
        # In real system, this would query the scheduler for available resources
        return [
            {"agent_id": "agent-alt-1", "capabilities": {"cpu_cores": 8, "memory_gb": 16}, "current_utilization": 0.3},
            {"agent_id": "agent-alt-2", "capabilities": {"cpu_cores": 16, "memory_gb": 32}, "current_utilization": 0.6},
            {"agent_id": "agent-alt-3", "capabilities": {"cpu_cores": 4, "memory_gb": 8}, "current_utilization": 0.1}
        ]
    
    def _calculate_average_wait_time(self) -> float:
        """Calculate average wait time for pending jobs"""
        if not self.pending_jobs:
            return 0.0
        
        current_time = datetime.now()
        total_wait = 0.0
        
        for job in self.pending_jobs:
            if hasattr(job, 'submit_time') and job.submit_time:
                wait_time = (current_time - job.submit_time).total_seconds()
                total_wait += wait_time
        
        return total_wait / len(self.pending_jobs) if self.pending_jobs else 0.0
    
    def update_performance_metrics(self, job_id: str, success: bool, completion_time: float):
        """Update performance metrics for LLM context"""
        if success:
            self.performance_metrics["successful_jobs"] += 1
        else:
            self.performance_metrics["failed_jobs"] += 1
        
        # Update running average of completion time
        total_jobs = self.performance_metrics["successful_jobs"] + self.performance_metrics["failed_jobs"]
        current_avg = self.performance_metrics["average_completion_time"]
        self.performance_metrics["average_completion_time"] = (
            (current_avg * (total_jobs - 1) + completion_time) / total_jobs
        )
        
        # Track utilization history
        self.performance_metrics["utilization_history"].append(self.resource.utilization_score)
        if len(self.performance_metrics["utilization_history"]) > 100:
            self.performance_metrics["utilization_history"] = self.performance_metrics["utilization_history"][-100:]
    
    def get_llm_status(self) -> Dict[str, Any]:
        """Get status of LLM integration"""
        return {
            "llm_enabled": self.llm_enabled,
            "llm_provider": str(self.llm_provider) if self.llm_provider else "default",
            "decisions_made": len(self.decision_history),
            "performance_metrics": self.performance_metrics,
            "context_window_size": len(self.context_builder.decision_history)
        }
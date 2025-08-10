#!/usr/bin/env python3
"""
Context Management for LLM-Enhanced Agents
==========================================

Builds rich context information for LLM decision-making in the
distributed multi-agent scheduling system.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from collections import deque, defaultdict

from ..jobs.job import Job, JobStatus, JobPriority
from ..resources.resource import Resource


@dataclass
class SystemMetrics:
    """Current system performance metrics"""
    total_jobs_submitted: int
    total_jobs_completed: int
    total_jobs_failed: int
    average_completion_time: float
    current_utilization: float
    available_resources: int
    busy_resources: int
    queue_length: int
    timestamp: datetime


@dataclass
class HistoricalPattern:
    """Historical system behavior patterns"""
    workload_patterns: Dict[str, Any]
    failure_patterns: Dict[str, Any]
    performance_trends: Dict[str, Any]
    resource_utilization_history: List[float]
    peak_hours: List[int]
    seasonal_patterns: Dict[str, Any]


class ContextBuilder:
    """Builds rich context for LLM decision-making"""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.job_history: deque = deque(maxlen=max_history_size)
        self.performance_history: deque = deque(maxlen=max_history_size)
        self.failure_history: deque = deque(maxlen=max_history_size)
        self.decision_history: deque = deque(maxlen=max_history_size)
        
    def build_job_scoring_context(self, 
                                 job: Union[Job, Dict],
                                 resource: Union[Resource, Dict],
                                 system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Build context for job scoring decisions"""
        
        job_data = job.to_dict() if hasattr(job, 'to_dict') else job
        resource_data = resource.to_dict() if hasattr(resource, 'to_dict') else resource
        
        context = {
            "job": {
                "id": job_data.get("job_id"),
                "name": job_data.get("name"),
                "priority": job_data.get("priority"),
                "requirements": job_data.get("requirements", {}),
                "estimated_runtime": job_data.get("requirements", {}).get("estimated_runtime_minutes", 0),
                "dependencies": job_data.get("dependencies", []),
                "user_id": job_data.get("user_id"),
                "submit_time": job_data.get("submit_time")
            },
            "resource": {
                "id": resource_data.get("resource_id"),
                "type": resource_data.get("type"),
                "capacity": resource_data.get("capacity", {}),
                "current_utilization": resource_data.get("current_utilization", 0),
                "available_cpu": resource_data.get("capacity", {}).get("total_cpu_cores", 0) * (1 - resource_data.get("current_utilization", 0)),
                "available_memory": resource_data.get("capacity", {}).get("total_memory_gb", 0) * (1 - resource_data.get("current_utilization", 0)),
                "location": resource_data.get("location"),
                "cost_per_hour": resource_data.get("cost_per_hour", 0)
            },
            "system_state": {
                "current_time": datetime.now().isoformat(),
                "queue_length": system_state.get("queue_length", 0),
                "average_wait_time": system_state.get("average_wait_time", 0),
                "system_load": system_state.get("system_load", 0.5),
                "available_agents": system_state.get("available_agents", 0),
                "busy_agents": system_state.get("busy_agents", 0)
            },
            "historical_performance": self._get_resource_performance_history(resource_data.get("resource_id")),
            "similar_jobs": self._get_similar_job_history(job_data),
            "objectives": {
                "minimize_wait_time": True,
                "maximize_utilization": True,
                "respect_priorities": True,
                "balance_load": True
            }
        }
        
        return context
    
    def build_fault_recovery_context(self,
                                   failure_info: Dict[str, Any],
                                   job: Union[Job, Dict],
                                   available_resources: List[Dict],
                                   system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Build context for fault recovery decisions"""
        
        job_data = job.to_dict() if hasattr(job, 'to_dict') else job
        
        context = {
            "failure": {
                "type": failure_info.get("type", "unknown"),
                "timestamp": failure_info.get("timestamp", datetime.now().isoformat()),
                "error_message": failure_info.get("error_message"),
                "failed_resource": failure_info.get("resource_id"),
                "retry_count": failure_info.get("retry_count", 0),
                "previous_attempts": failure_info.get("previous_attempts", [])
            },
            "job": {
                "id": job_data.get("job_id"),
                "priority": job_data.get("priority"),
                "requirements": job_data.get("requirements", {}),
                "deadline": job_data.get("deadline"),
                "criticality": self._assess_job_criticality(job_data),
                "retry_tolerance": job_data.get("retry_tolerance", 3)
            },
            "available_resources": [
                {
                    "id": res.get("agent_id"),
                    "capacity": res.get("capabilities", {}),
                    "current_load": res.get("current_utilization", 0),
                    "reliability_score": self._get_resource_reliability(res.get("agent_id")),
                    "distance_from_failed": self._calculate_resource_distance(
                        res.get("agent_id"), failure_info.get("resource_id")
                    )
                }
                for res in available_resources
            ],
            "system_state": {
                "current_time": datetime.now().isoformat(),
                "failure_rate": system_state.get("recent_failure_rate", 0.05),
                "system_stress": system_state.get("system_stress", 0.5),
                "recovery_resources_available": len(available_resources)
            },
            "failure_patterns": self._analyze_failure_patterns(),
            "recovery_strategies": {
                "immediate_retry": True,
                "alternative_resource": len(available_resources) > 0,
                "delayed_retry": True,
                "job_modification": False,
                "human_escalation": failure_info.get("retry_count", 0) > 5
            }
        }
        
        return context
    
    def build_negotiation_context(self,
                                 proposal: Dict[str, Any],
                                 agent_state: Dict[str, Any],
                                 market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Build context for negotiation decisions"""
        
        context = {
            "proposal": {
                "job_id": proposal.get("job_id"),
                "resource_request": proposal.get("resource_request", {}),
                "priority": proposal.get("priority", 2),
                "estimated_duration": proposal.get("estimated_duration", 0),
                "deadline": proposal.get("deadline"),
                "offered_price": proposal.get("offered_price", 0),
                "sender": proposal.get("sender_id"),
                "negotiation_round": proposal.get("negotiation_round", 1)
            },
            "agent_state": {
                "current_utilization": agent_state.get("utilization", 0),
                "available_capacity": agent_state.get("available_capacity", {}),
                "scheduled_jobs": agent_state.get("scheduled_jobs", []),
                "revenue_target": agent_state.get("revenue_target", 0),
                "reputation_score": agent_state.get("reputation_score", 0.5),
                "strategic_goals": agent_state.get("strategic_goals", [])
            },
            "market_conditions": {
                "demand_level": market_conditions.get("demand_level", "medium"),
                "supply_availability": market_conditions.get("supply_availability", "medium"),
                "average_prices": market_conditions.get("average_prices", {}),
                "competitor_activity": market_conditions.get("competitor_activity", {}),
                "peak_hours": market_conditions.get("is_peak_hour", False)
            },
            "negotiation_history": self._get_negotiation_history(proposal.get("sender_id")),
            "strategic_considerations": {
                "long_term_partnership": self._assess_partnership_potential(proposal.get("sender_id")),
                "reputation_impact": self._assess_reputation_impact(proposal),
                "opportunity_cost": self._calculate_opportunity_cost(proposal, agent_state)
            }
        }
        
        return context
    
    def build_scheduling_context(self,
                               pending_jobs: List[Union[Job, Dict]],
                               available_resources: List[Dict],
                               system_objectives: Dict[str, Any]) -> Dict[str, Any]:
        """Build context for scheduling decisions"""
        
        jobs_data = []
        for job in pending_jobs:
            job_dict = job.to_dict() if hasattr(job, 'to_dict') else job
            jobs_data.append({
                "id": job_dict.get("job_id"),
                "priority": job_dict.get("priority"),
                "requirements": job_dict.get("requirements", {}),
                "submit_time": job_dict.get("submit_time"),
                "dependencies": job_dict.get("dependencies", []),
                "estimated_runtime": job_dict.get("requirements", {}).get("estimated_runtime_minutes", 0),
                "user_id": job_dict.get("user_id"),
                "criticality": self._assess_job_criticality(job_dict)
            })
        
        context = {
            "pending_jobs": jobs_data,
            "available_resources": [
                {
                    "id": res.get("agent_id"),
                    "type": res.get("type", "compute"),
                    "capabilities": res.get("capabilities", {}),
                    "current_utilization": res.get("current_utilization", 0),
                    "performance_score": self._get_resource_performance_score(res.get("agent_id")),
                    "cost_efficiency": self._get_cost_efficiency_score(res.get("agent_id")),
                    "location": res.get("location"),
                    "specializations": res.get("specializations", [])
                }
                for res in available_resources
            ],
            "system_objectives": {
                "throughput_weight": system_objectives.get("throughput_weight", 0.3),
                "fairness_weight": system_objectives.get("fairness_weight", 0.2),
                "efficiency_weight": system_objectives.get("efficiency_weight", 0.3),
                "cost_weight": system_objectives.get("cost_weight", 0.2),
                "deadline_compliance": system_objectives.get("deadline_compliance", True),
                "load_balancing": system_objectives.get("load_balancing", True)
            },
            "constraints": {
                "resource_limits": system_objectives.get("resource_limits", {}),
                "priority_thresholds": system_objectives.get("priority_thresholds", {}),
                "deadline_constraints": self._extract_deadline_constraints(jobs_data),
                "dependency_constraints": self._extract_dependency_constraints(jobs_data)
            },
            "current_state": {
                "timestamp": datetime.now().isoformat(),
                "queue_length": len(pending_jobs),
                "average_wait_time": self._calculate_current_wait_time(),
                "system_load": self._calculate_system_load(available_resources),
                "resource_diversity": len(set(res.get("type") for res in available_resources))
            },
            "optimization_context": {
                "workload_patterns": self._get_current_workload_patterns(),
                "performance_predictions": self._get_performance_predictions(),
                "resource_forecasts": self._get_resource_availability_forecasts()
            }
        }
        
        return context
    
    def record_decision(self, decision_type: str, context: Dict[str, Any], 
                       decision: Dict[str, Any], outcome: Optional[Dict[str, Any]] = None):
        """Record a decision for future learning"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "decision_type": decision_type,
            "context_summary": self._summarize_context(context),
            "decision": decision,
            "outcome": outcome
        }
        self.decision_history.append(record)
    
    def _get_resource_performance_history(self, resource_id: str) -> Dict[str, Any]:
        """Get performance history for a specific resource"""
        # Mock implementation - in real system, query historical data
        return {
            "average_completion_time": 150.0,
            "success_rate": 0.95,
            "utilization_trend": "stable",
            "recent_failures": 2,
            "performance_score": 0.87
        }
    
    def _get_similar_job_history(self, job_data: Dict) -> List[Dict]:
        """Find similar jobs from history"""
        # Mock implementation - in real system, use ML similarity matching
        return [
            {
                "job_id": "similar-1",
                "completion_time": 120,
                "success": True,
                "resource_used": "agent-2"
            },
            {
                "job_id": "similar-2", 
                "completion_time": 180,
                "success": True,
                "resource_used": "agent-1"
            }
        ]
    
    def _assess_job_criticality(self, job_data: Dict) -> str:
        """Assess the criticality level of a job"""
        priority = job_data.get("priority", 2)
        has_deadline = job_data.get("deadline") is not None
        
        if priority >= 4 or has_deadline:
            return "critical"
        elif priority >= 3:
            return "high"
        elif priority >= 2:
            return "medium"
        else:
            return "low"
    
    def _get_resource_reliability(self, resource_id: str) -> float:
        """Get reliability score for a resource"""
        # Mock implementation - in real system, calculate from failure history
        return 0.92
    
    def _calculate_resource_distance(self, resource1_id: str, resource2_id: str) -> str:
        """Calculate logical distance between resources"""
        # Mock implementation - in real system, use network topology
        if resource1_id == resource2_id:
            return "same"
        elif abs(hash(resource1_id) - hash(resource2_id)) % 3 == 0:
            return "near"
        else:
            return "far"
    
    def _analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze recent failure patterns"""
        return {
            "most_common_failure": "resource_exhaustion",
            "failure_correlation": "high_load_periods",
            "recovery_success_rate": 0.88,
            "time_to_recovery": 45.0
        }
    
    def _get_negotiation_history(self, partner_id: str) -> Dict[str, Any]:
        """Get negotiation history with a specific partner"""
        return {
            "successful_negotiations": 15,
            "failed_negotiations": 3,
            "average_rounds": 2.3,
            "satisfaction_score": 0.82
        }
    
    def _assess_partnership_potential(self, partner_id: str) -> str:
        """Assess long-term partnership potential"""
        return "high"  # Mock implementation
    
    def _assess_reputation_impact(self, proposal: Dict) -> str:
        """Assess reputation impact of accepting/rejecting proposal"""
        return "neutral"  # Mock implementation
    
    def _calculate_opportunity_cost(self, proposal: Dict, agent_state: Dict) -> float:
        """Calculate opportunity cost of accepting proposal"""
        return 0.15  # Mock implementation
    
    def _get_resource_performance_score(self, resource_id: str) -> float:
        """Get performance score for a resource"""
        return 0.85  # Mock implementation
    
    def _get_cost_efficiency_score(self, resource_id: str) -> float:
        """Get cost efficiency score for a resource"""
        return 0.78  # Mock implementation
    
    def _extract_deadline_constraints(self, jobs_data: List[Dict]) -> List[Dict]:
        """Extract deadline constraints from jobs"""
        return [
            {"job_id": job["id"], "deadline": job.get("deadline")}
            for job in jobs_data if job.get("deadline")
        ]
    
    def _extract_dependency_constraints(self, jobs_data: List[Dict]) -> List[Dict]:
        """Extract dependency constraints from jobs"""
        return [
            {"job_id": job["id"], "dependencies": job.get("dependencies", [])}
            for job in jobs_data if job.get("dependencies")
        ]
    
    def _calculate_current_wait_time(self) -> float:
        """Calculate current average wait time"""
        return 75.0  # Mock implementation
    
    def _calculate_system_load(self, resources: List[Dict]) -> float:
        """Calculate overall system load"""
        if not resources:
            return 0.0
        total_utilization = sum(res.get("current_utilization", 0) for res in resources)
        return total_utilization / len(resources)
    
    def _get_current_workload_patterns(self) -> Dict[str, Any]:
        """Get current workload patterns"""
        return {
            "trend": "increasing",
            "peak_expected": False,
            "job_type_distribution": {"cpu_intensive": 0.6, "memory_intensive": 0.4}
        }
    
    def _get_performance_predictions(self) -> Dict[str, Any]:
        """Get performance predictions"""
        return {
            "expected_completion_times": {"cpu_intensive": 120, "memory_intensive": 90},
            "resource_availability": {"next_hour": 0.8, "next_4_hours": 0.6}
        }
    
    def _get_resource_availability_forecasts(self) -> Dict[str, Any]:
        """Get resource availability forecasts"""
        return {
            "predicted_free_resources": {"1_hour": 3, "4_hours": 7},
            "maintenance_windows": [],
            "expected_failures": 0.1
        }
    
    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the context for storage"""
        return {
            "context_type": context.get("decision_type", "unknown"),
            "key_factors": list(context.keys())[:5],
            "timestamp": datetime.now().isoformat()
        }
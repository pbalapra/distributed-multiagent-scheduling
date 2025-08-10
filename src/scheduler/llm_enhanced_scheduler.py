#!/usr/bin/env python3
"""
LLM-Enhanced Discrete Event Scheduler
=====================================

Extends the discrete event scheduler with LLM capabilities for
intelligent scheduling decisions, resource allocation optimization,
and predictive system management.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from .discrete_event_scheduler import DiscreteEventScheduler, SchedulingEvent, SchedulingEventType
from ..llm.llm_interface import LLMRequest, llm_manager, LLMProvider
from ..llm.context_manager import ContextBuilder
from ..jobs.job import Job, JobStatus


class LLMEnhancedScheduler(DiscreteEventScheduler):
    """Discrete event scheduler with LLM-enhanced decision making"""
    
    def __init__(self, message_bus, job_pool,
                 llm_enabled: bool = True,
                 llm_provider: Optional[LLMProvider] = None,
                 predictive_scheduling: bool = True):
        """
        Initialize LLM-enhanced scheduler
        
        Args:
            message_bus: Communication bus for agents
            job_pool: Pool of jobs to schedule
            llm_enabled: Whether to use LLM for scheduling decisions
            llm_provider: Specific LLM provider to use
            predictive_scheduling: Enable predictive scheduling features
        """
        super().__init__(message_bus, job_pool)
        
        self.llm_enabled = llm_enabled
        self.llm_provider = llm_provider
        self.predictive_scheduling = predictive_scheduling
        self.context_builder = ContextBuilder()
        
        # Enhanced tracking for LLM context
        self.scheduling_history: List[Dict] = []
        self.system_predictions: Dict[str, Any] = {}
        self.optimization_objectives = {
            "throughput_weight": 0.3,
            "fairness_weight": 0.2,
            "efficiency_weight": 0.3,
            "cost_weight": 0.2
        }
        
        # LLM configuration for different decision types
        self.llm_config = {
            "scheduling": {
                "temperature": 0.1,
                "max_tokens": 1000,
                "timeout": 20.0
            },
            "resource_allocation": {
                "temperature": 0.15,
                "max_tokens": 800,
                "timeout": 15.0
            },
            "failure_prediction": {
                "temperature": 0.05,
                "max_tokens": 600,
                "timeout": 10.0
            }
        }
        
        print(f"🧠 LLM-Enhanced Scheduler initialized "
              f"(LLM: {'enabled' if llm_enabled else 'disabled'}, "
              f"Predictive: {'enabled' if predictive_scheduling else 'disabled'})")
    
    def _handle_scheduling_decision(self, event: SchedulingEvent):
        """Enhanced scheduling decision with LLM reasoning"""
        
        if not self.llm_enabled:
            return super()._handle_scheduling_decision(event)
        
        try:
            trigger = event.data.get('trigger', 'unknown')
            print(f"🧠 Making LLM-enhanced scheduling decision (triggered by: {trigger})")
            
            # Get system state
            pending_jobs = self.job_pool.get_pending_jobs()
            available_resources = self._get_available_resources()
            
            if not pending_jobs or not available_resources:
                print("⚠️ No jobs or resources available for LLM scheduling")
                return super()._handle_scheduling_decision(event)
            
            # Build rich context for LLM
            context = self.context_builder.build_scheduling_context(
                pending_jobs, available_resources, self.optimization_objectives
            )
            
            # Create LLM request
            prompt = self._build_scheduling_prompt(context)
            request = LLMRequest(
                prompt=prompt,
                context=context,
                task_type="scheduling",
                **self.llm_config["scheduling"]
            )
            
            # Get LLM response
            response = llm_manager.generate_sync(request, self.llm_provider)
            scheduling_decision = json.loads(response.content)
            
            # Execute LLM scheduling decision
            assignments_made = self._execute_llm_scheduling(scheduling_decision, pending_jobs, available_resources)
            
            # Record decision for learning
            outcome = {"assignments_made": assignments_made, "timestamp": datetime.now().isoformat()}
            self.context_builder.record_decision("scheduling", context, scheduling_decision, outcome)
            
            print(f"🎯 LLM made {assignments_made} job assignments")
            if response.reasoning:
                print(f"   Reasoning: {response.reasoning[:150]}...")
            
        except Exception as e:
            print(f"⚠️ LLM scheduling failed: {e}")
            print("🔄 Falling back to heuristic scheduling")
            super()._handle_scheduling_decision(event)
    
    def _handle_resource_failure(self, event: SchedulingEvent):
        """Enhanced resource failure handling with LLM prediction"""
        
        if self.predictive_scheduling and self.llm_enabled:
            try:
                # Use LLM to predict cascading failures and preemptive actions
                self._predict_and_prevent_cascading_failures(event)
            except Exception as e:
                print(f"⚠️ LLM failure prediction failed: {e}")
        
        # Execute standard failure handling
        super()._handle_resource_failure(event)
    
    def _predict_and_prevent_cascading_failures(self, failure_event: SchedulingEvent):
        """Use LLM to predict and prevent cascading failures"""
        
        failed_agent = failure_event.data.get('agent_id')
        system_state = self._get_comprehensive_system_state()
        
        # Build prediction context
        context = {
            "failure_event": {
                "failed_resource": failed_agent,
                "timestamp": failure_event.timestamp.isoformat(),
                "system_load_at_failure": system_state.get("system_load", 0.5)
            },
            "system_state": system_state,
            "resource_dependencies": self._analyze_resource_dependencies(),
            "current_workload": self._analyze_current_workload(),
            "historical_failures": self._get_failure_history()
        }
        
        # Create LLM request for failure prediction
        prompt = self._build_failure_prediction_prompt(context)
        request = LLMRequest(
            prompt=prompt,
            context=context,
            task_type="failure_prediction",
            **self.llm_config["failure_prediction"]
        )
        
        # Get LLM prediction
        response = llm_manager.generate_sync(request, self.llm_provider)
        prediction = json.loads(response.content)
        
        # Execute preventive actions if needed
        if prediction.get("cascading_risk", 0.0) > 0.7:
            self._execute_preventive_actions(prediction)
            print(f"🛡️ Executed preventive actions based on LLM prediction")
            if response.reasoning:
                print(f"   Reasoning: {response.reasoning[:150]}...")
    
    def _execute_llm_scheduling(self, decision: Dict[str, Any], 
                               pending_jobs: List[Job], 
                               available_resources: List[Dict]) -> int:
        """Execute scheduling decision from LLM"""
        
        assignments = decision.get("assignments", [])
        assignments_made = 0
        
        # Create mapping for quick lookup
        job_map = {job.job_id: job for job in pending_jobs}
        resource_map = {res["agent_id"]: res for res in available_resources}
        
        for assignment in assignments:
            job_id = assignment.get("job_id")
            resource_id = assignment.get("resource_id")
            
            if job_id in job_map and resource_id in resource_map:
                job = job_map[job_id]
                
                # Check if resource is still available
                if resource_id in self.available_agents:
                    self._assign_job_to_agent(job, resource_id, datetime.now())
                    assignments_made += 1
                else:
                    print(f"⚠️ Resource {resource_id} no longer available for job {job_id}")
        
        return assignments_made
    
    def _execute_preventive_actions(self, prediction: Dict[str, Any]):
        """Execute preventive actions based on LLM predictions"""
        
        actions = prediction.get("preventive_actions", [])
        
        for action in actions:
            action_type = action.get("type")
            
            if action_type == "redistribute_load":
                self._redistribute_workload(action.get("target_resources", []))
            elif action_type == "increase_monitoring":
                self._increase_monitoring_frequency(action.get("resources", []))
            elif action_type == "prepare_backup":
                self._prepare_backup_resources(action.get("backup_count", 1))
            elif action_type == "notify_operators":
                self._notify_human_operators(action.get("message", "Potential system issues predicted"))
    
    def _get_available_resources(self) -> List[Dict]:
        """Get detailed information about available resources"""
        resources = []
        
        for agent_id, agent_info in self.available_agents.items():
            resource_info = {
                "agent_id": agent_id,
                "type": "compute",  # Mock - in real system, get from agent
                "capabilities": agent_info.get("capabilities", {}),
                "current_utilization": agent_info.get("utilization", 0.0),
                "performance_score": 0.85,  # Mock - calculate from history
                "cost_efficiency": 0.78,   # Mock - calculate from metrics
                "location": f"datacenter-{agent_id[-1]}",
                "specializations": ["cpu_intensive", "parallel_processing"]
            }
            resources.append(resource_info)
        
        return resources
    
    def _get_comprehensive_system_state(self) -> Dict[str, Any]:
        """Get comprehensive system state for LLM context"""
        stats = self.get_system_stats()
        
        return {
            "current_time": datetime.now().isoformat(),
            "system_load": stats["available_agents"] / max(1, stats["available_agents"] + stats["busy_agents"]),
            "queue_length": stats.get("pending_jobs", 0),
            "active_jobs": stats["active_assignments"],
            "recent_failure_rate": 0.05,  # Mock - calculate from recent history
            "resource_utilization": 0.75,  # Mock - calculate average utilization
            "performance_trend": "stable"  # Mock - analyze recent performance
        }
    
    def _analyze_resource_dependencies(self) -> Dict[str, Any]:
        """Analyze dependencies between resources"""
        # Mock implementation - in real system, analyze resource relationships
        return {
            "dependency_graph": {"agent-1": ["agent-2"], "agent-3": ["agent-4"]},
            "critical_resources": ["agent-1", "agent-3"],
            "isolation_groups": [["agent-1", "agent-2"], ["agent-3", "agent-4"]]
        }
    
    def _analyze_current_workload(self) -> Dict[str, Any]:
        """Analyze current workload characteristics"""
        return {
            "total_jobs": len(self.job_assignments),
            "job_type_distribution": {"cpu_intensive": 0.6, "memory_intensive": 0.4},
            "priority_distribution": {"high": 0.3, "medium": 0.5, "low": 0.2},
            "estimated_completion_time": 180  # Mock average
        }
    
    def _get_failure_history(self) -> List[Dict]:
        """Get recent failure history"""
        # Mock implementation - in real system, query failure database
        return [
            {"timestamp": "2024-01-01T10:00:00", "resource": "agent-2", "type": "hardware"},
            {"timestamp": "2024-01-01T14:30:00", "resource": "agent-1", "type": "software"}
        ]
    
    def _build_scheduling_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for LLM scheduling decisions"""
        
        jobs = context["pending_jobs"]
        resources = context["available_resources"]
        objectives = context["system_objectives"]
        constraints = context["constraints"]
        
        prompt = f"""
You are an intelligent HPC job scheduler optimizing resource allocation.

TASK: Create optimal job-to-resource assignments considering multiple objectives.

PENDING JOBS ({len(jobs)}):
{json.dumps(jobs[:5], indent=2)}
{'... (showing first 5 jobs)' if len(jobs) > 5 else ''}

AVAILABLE RESOURCES ({len(resources)}):
{json.dumps(resources[:3], indent=2)}
{'... (showing first 3 resources)' if len(resources) > 3 else ''}

OPTIMIZATION OBJECTIVES:
- Throughput Weight: {objectives['throughput_weight']}
- Fairness Weight: {objectives['fairness_weight']}  
- Efficiency Weight: {objectives['efficiency_weight']}
- Cost Weight: {objectives['cost_weight']}

CONSTRAINTS:
- Deadline Jobs: {len(constraints['deadline_constraints'])}
- Dependencies: {len(constraints['dependency_constraints'])}
- Resource Limits: {bool(constraints['resource_limits'])}

STRATEGIC GOALS:
- Maximize system throughput
- Ensure fair resource allocation
- Respect job priorities and deadlines
- Optimize resource utilization
- Minimize total execution cost

Return JSON response with optimal assignments:
{{
  "assignments": [
    {{"job_id": "<job_id>", "resource_id": "<resource_id>", "priority_score": <float>, "rationale": "<brief_reason>"}},
    ...
  ],
  "strategy": "<scheduling_strategy_name>",
  "load_balancing": "<distribution_approach>",
  "optimization_focus": "<primary_objective>"
}}

Consider job requirements, resource capabilities, current load, and strategic objectives.
"""
        return prompt
    
    def _build_failure_prediction_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for failure prediction"""
        
        failure = context["failure_event"]
        system = context["system_state"]
        
        prompt = f"""
You are a predictive system analyst monitoring HPC infrastructure for potential cascading failures.

TASK: Assess cascading failure risk and recommend preventive actions.

CURRENT FAILURE:
- Failed Resource: {failure['failed_resource']}
- System Load: {failure['system_load_at_failure']:.2f}
- Timestamp: {failure['timestamp']}

SYSTEM STATE:
- Current Load: {system['system_load']:.2f}
- Queue Length: {system['queue_length']}
- Active Jobs: {system['active_jobs']}
- Recent Failure Rate: {system['recent_failure_rate']:.3f}

RESOURCE DEPENDENCIES:
{json.dumps(context.get('resource_dependencies', {}), indent=2)}

RISK FACTORS:
- High system load (>0.8)
- Resource dependencies
- Recent failure patterns
- Current workload stress

Return JSON assessment:
{{
  "cascading_risk": <float 0.0-1.0>,
  "risk_factors": ["factor1", "factor2"],
  "affected_resources": ["resource1", "resource2"],
  "preventive_actions": [
    {{"type": "redistribute_load", "target_resources": ["res1"], "priority": "high"}},
    {{"type": "increase_monitoring", "resources": ["res2"], "priority": "medium"}}
  ],
  "time_window": "<critical_period>",
  "confidence": <float 0.0-1.0>
}}

Focus on preventing cascading failures and maintaining system stability.
"""
        return prompt
    
    def _redistribute_workload(self, target_resources: List[str]):
        """Redistribute workload to prevent overload"""
        print(f"🔄 Redistributing workload from {len(target_resources)} resources")
        # Implementation would move jobs between resources
    
    def _increase_monitoring_frequency(self, resources: List[str]):
        """Increase monitoring frequency for at-risk resources"""
        print(f"👁️ Increasing monitoring for {len(resources)} resources")
        # Implementation would adjust monitoring parameters
    
    def _prepare_backup_resources(self, backup_count: int):
        """Prepare backup resources for potential failures"""
        print(f"🔒 Preparing {backup_count} backup resources")
        # Implementation would provision standby resources
    
    def _notify_human_operators(self, message: str):
        """Notify human operators of potential issues"""
        print(f"📢 Operator notification: {message}")
        # Implementation would send alerts to operators
    
    def get_llm_scheduler_status(self) -> Dict[str, Any]:
        """Get LLM scheduler status and metrics"""
        return {
            "llm_enabled": self.llm_enabled,
            "predictive_scheduling": self.predictive_scheduling,
            "llm_provider": str(self.llm_provider) if self.llm_provider else "default",
            "scheduling_decisions": len(self.scheduling_history),
            "optimization_objectives": self.optimization_objectives,
            "system_predictions": self.system_predictions,
            "base_stats": self.get_system_stats()
        }
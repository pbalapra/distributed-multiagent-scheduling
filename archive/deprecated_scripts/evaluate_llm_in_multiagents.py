#!/usr/bin/env python3
"""
LLM Evaluation Methods in Multiagent Systems
===========================================

This script demonstrates multiple approaches to evaluate LLM performance
within the distributed multiagent scheduling system.
"""

import sys
import time
import json
import statistics
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.ollama_provider import OllamaProvider
from llm.llm_interface import LLMRequest, LLMManager, LLMProvider
from agents.llm_resource_agent import LLMResourceAgent
from resources.resource import Resource, ResourceCapacity, ResourceType
from jobs.job import Job, JobPriority, ResourceRequirement
from communication.protocol import MessageBus

class LLMMultiagentEvaluator:
    """Comprehensive LLM evaluation in multiagent context"""
    
    def __init__(self):
        self.provider = OllamaProvider(model_name="mistral", timeout=60.0)
        self.results = {
            "response_times": [],
            "token_counts": [],
            "decision_quality": [],
            "json_validity": [],
            "agent_performance": []
        }
    
    def evaluate_llm_integration(self):
        """1. Evaluate basic LLM integration"""
        print("🔍 1. BASIC LLM INTEGRATION EVALUATION")
        print("-" * 50)
        
        tests = [
            ("Simple Query", "What is 2+2?", "basic", 10),
            ("Complex Analysis", "Analyze resource allocation for high-priority job", "analysis", 50),
            ("JSON Generation", "Provide job scoring in JSON format", "json", 100)
        ]
        
        for name, prompt, task_type, max_tokens in tests:
            request = LLMRequest(
                prompt=prompt,
                context={},
                task_type=task_type,
                max_tokens=max_tokens
            )
            
            start_time = time.time()
            response = self.provider.generate_sync(request)
            response_time = time.time() - start_time
            
            # Collect metrics
            self.results["response_times"].append(response_time)
            self.results["token_counts"].append(response.metadata.get("tokens_used", 0))
            
            # Check JSON validity for structured responses
            is_valid_json = False
            if task_type == "json":
                try:
                    json.loads(response.content)
                    is_valid_json = True
                except:
                    pass
            self.results["json_validity"].append(is_valid_json)
            
            print(f"✅ {name}: {response_time:.2f}s, {response.metadata.get('tokens_used', 0)} tokens")
            if is_valid_json:
                print(f"   📄 Valid JSON response generated")
    
    def evaluate_agent_decision_making(self):
        """2. Evaluate LLM-enhanced agent decision making"""
        print("\n🤖 2. AGENT DECISION MAKING EVALUATION")
        print("-" * 50)
        
        # Create test environment
        message_bus = MessageBus()
        
        # Create resource
        capacity = ResourceCapacity(
            total_cpu_cores=16,
            total_memory_gb=64,
            total_gpu_count=2,
            total_storage_gb=1000
        )
        resource = Resource(
            resource_id="test-resource",
            name="Test HPC Cluster",
            resource_type=ResourceType.CPU_CLUSTER,
            capacity=capacity,
            location="TestCenter",
            cost_per_hour=2.50
        )
        
        # Create LLM-enabled agent
        agent = LLMResourceAgent(
            agent_id="test-llm-agent",
            resource=resource,
            message_bus=message_bus,
            llm_enabled=True
        )
        
        # Test different job scenarios
        test_jobs = [
            ("Small Job", 2, 4, JobPriority.LOW),
            ("Medium Job", 8, 16, JobPriority.MEDIUM), 
            ("Large Job", 16, 32, JobPriority.HIGH),
            ("Critical Job", 12, 24, JobPriority.CRITICAL)
        ]
        
        decision_scores = []
        
        for job_name, cpu, memory, priority in test_jobs:
            # Create test job
            requirements = ResourceRequirement(
                cpu_cores=cpu,
                memory_gb=memory,
                gpu_count=0,
                estimated_runtime_minutes=60
            )
            
            job = Job(
                job_id=f"test-job-{cpu}c-{memory}gb",
                name=job_name,
                user_id="test_user",
                requirements=requirements,
                priority=priority,
                command="test_command",
                working_directory="/test",
                submit_time=datetime.now()
            )
            
            # Get LLM-enhanced job score
            start_time = time.time()
            job_data = job.to_dict()
            llm_score = agent._calculate_job_score(job_data)
            decision_time = time.time() - start_time
            
            # Get heuristic score for comparison
            agent.llm_enabled = False
            heuristic_score = agent._calculate_job_score(job_data)
            agent.llm_enabled = True
            
            decision_scores.append({
                "job": job_name,
                "llm_score": llm_score,
                "heuristic_score": heuristic_score,
                "decision_time": decision_time,
                "improvement": llm_score - heuristic_score
            })
            
            print(f"📊 {job_name}: LLM={llm_score:.3f}, Heuristic={heuristic_score:.3f}, "
                  f"Improvement={llm_score-heuristic_score:+.3f}, Time={decision_time:.2f}s")
        
        self.results["agent_performance"] = decision_scores
    
    def evaluate_fault_recovery_intelligence(self):
        """3. Evaluate LLM fault recovery intelligence"""
        print("\n🛠️ 3. FAULT RECOVERY INTELLIGENCE EVALUATION")
        print("-" * 50)
        
        # Create agent for fault recovery testing
        message_bus = MessageBus()
        capacity = ResourceCapacity(16, 64, 2, 1000)
        resource = Resource("recovery-test", "Recovery Test", ResourceType.CPU_CLUSTER, capacity, "TestCenter", 2.50)
        agent = LLMResourceAgent("recovery-agent", resource, message_bus, llm_enabled=True)
        
        # Test various failure scenarios
        failure_scenarios = [
            {
                "name": "Network Timeout",
                "failure_info": {
                    "type": "network_timeout",
                    "retry_count": 0,
                    "error_message": "Connection timeout",
                    "resource_id": "worker-1",
                    "timestamp": datetime.now().isoformat()
                }
            },
            {
                "name": "Resource Exhaustion", 
                "failure_info": {
                    "type": "resource_exhaustion",
                    "retry_count": 1,
                    "error_message": "Out of memory",
                    "resource_id": "worker-2",
                    "timestamp": datetime.now().isoformat()
                }
            },
            {
                "name": "Hardware Failure",
                "failure_info": {
                    "type": "hardware_failure", 
                    "retry_count": 3,
                    "error_message": "GPU device error",
                    "resource_id": "worker-3",
                    "timestamp": datetime.now().isoformat()
                }
            }
        ]
        
        recovery_decisions = []
        
        for scenario in failure_scenarios:
            # Create test job
            job = Job(
                job_id=f"recovery-test-{scenario['name'].lower().replace(' ', '-')}",
                name=f"Recovery Test {scenario['name']}",
                user_id="test_user",
                requirements=ResourceRequirement(4, 8, 0, 120),
                priority=JobPriority.HIGH,
                command="recovery_test",
                working_directory="/test",
                submit_time=datetime.now()
            )
            agent.pending_jobs = [job]
            
            start_time = time.time()
            recovery_strategy = agent._handle_job_failure(job.job_id, scenario["failure_info"])
            decision_time = time.time() - start_time
            
            recovery_decisions.append({
                "scenario": scenario["name"],
                "strategy": recovery_strategy.get("strategy", "unknown"),
                "action": recovery_strategy.get("action", "none"),
                "decision_time": decision_time,
                "has_reasoning": "reasoning" in recovery_strategy
            })
            
            print(f"🚨 {scenario['name']}: Strategy={recovery_strategy.get('strategy', 'unknown')}, "
                  f"Action={recovery_strategy.get('action', 'none')}, Time={decision_time:.2f}s")
    
    def evaluate_negotiation_intelligence(self):
        """4. Evaluate LLM negotiation intelligence"""
        print("\n🤝 4. NEGOTIATION INTELLIGENCE EVALUATION") 
        print("-" * 50)
        
        # Create agent for negotiation testing
        message_bus = MessageBus()
        capacity = ResourceCapacity(16, 32, 1, 500)
        resource = Resource("negotiator", "Negotiator Agent", ResourceType.CPU_CLUSTER, capacity, "TestCenter", 3.00)
        
        # Test different utilization levels
        utilization_levels = [0.3, 0.7, 0.9]
        negotiation_results = []
        
        for utilization in utilization_levels:
            # Set utilization level
            resource.utilization.used_cpu_cores = int(resource.capacity.total_cpu_cores * utilization)
            resource.utilization.used_memory_gb = resource.capacity.total_memory_gb * utilization
            
            agent = LLMResourceAgent(f"negotiator-{int(utilization*100)}", resource, message_bus, llm_enabled=True)
            
            # Create negotiation proposal
            proposal = {
                "job_id": f"negotiation-test-{int(utilization*100)}",
                "resource_request": {"cpu_cores": 6, "memory_gb": 12},
                "priority": 3,
                "estimated_duration": 90,
                "offered_price": 8.00,
                "sender_id": "test-client",
                "negotiation_round": 1
            }
            
            start_time = time.time()
            decision = agent._evaluate_proposal(proposal)
            decision_time = time.time() - start_time
            
            negotiation_results.append({
                "utilization": utilization,
                "accept": decision.get("accept", False),
                "has_counter_offer": "counter_offer" in decision and decision["counter_offer"] is not None,
                "decision_time": decision_time,
                "has_reasoning": "reasoning" in decision or "reasoning_factors" in decision
            })
            
            status = "ACCEPT" if decision.get("accept", False) else "REJECT"
            counter = "Yes" if decision.get("counter_offer") else "No"
            print(f"📊 Utilization {utilization:.0%}: {status}, Counter-offer: {counter}, Time: {decision_time:.2f}s")
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        print("\n📊 LLM MULTIAGENT EVALUATION REPORT")
        print("=" * 60)
        
        # Response time analysis
        if self.results["response_times"]:
            avg_response_time = statistics.mean(self.results["response_times"])
            max_response_time = max(self.results["response_times"])
            min_response_time = min(self.results["response_times"])
            
            print(f"\n⏱️ RESPONSE TIME ANALYSIS:")
            print(f"   Average: {avg_response_time:.2f}s")
            print(f"   Range: {min_response_time:.2f}s - {max_response_time:.2f}s")
        
        # Token usage analysis
        if self.results["token_counts"]:
            avg_tokens = statistics.mean(self.results["token_counts"])
            total_tokens = sum(self.results["token_counts"])
            
            print(f"\n🔤 TOKEN USAGE ANALYSIS:")
            print(f"   Average per request: {avg_tokens:.1f} tokens")
            print(f"   Total tokens: {total_tokens}")
        
        # JSON validity
        json_valid_count = sum(self.results["json_validity"])
        json_total = len(self.results["json_validity"])
        if json_total > 0:
            json_success_rate = (json_valid_count / json_total) * 100
            print(f"\n📋 JSON RESPONSE QUALITY:")
            print(f"   Valid JSON responses: {json_valid_count}/{json_total} ({json_success_rate:.1f}%)")
        
        # Agent performance analysis
        if self.results["agent_performance"]:
            improvements = [d["improvement"] for d in self.results["agent_performance"]]
            avg_improvement = statistics.mean(improvements)
            positive_improvements = sum(1 for imp in improvements if imp > 0)
            
            print(f"\n🤖 AGENT DECISION QUALITY:")
            print(f"   Average LLM improvement: {avg_improvement:+.3f}")
            print(f"   Positive improvements: {positive_improvements}/{len(improvements)}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL LLM MULTIAGENT ASSESSMENT:")
        
        # Calculate overall score
        score_factors = []
        
        # Response time score (faster is better, target < 10s)
        if self.results["response_times"]:
            avg_time = statistics.mean(self.results["response_times"])
            time_score = max(0, min(100, 100 - (avg_time - 5) * 10))  # 100 at 5s, decreasing
            score_factors.append(("Response Time", time_score))
        
        # JSON validity score
        if json_total > 0:
            json_score = json_success_rate
            score_factors.append(("JSON Quality", json_score))
        
        # Decision improvement score
        if self.results["agent_performance"]:
            improvement_score = max(0, min(100, 50 + avg_improvement * 500))  # Scale improvements
            score_factors.append(("Decision Quality", improvement_score))
        
        if score_factors:
            overall_score = statistics.mean([score for _, score in score_factors])
            
            print(f"   Overall Score: {overall_score:.1f}/100")
            
            if overall_score >= 80:
                print("   🎉 EXCELLENT: LLM integration exceeds expectations")
                print("   - Fast response times with high-quality decisions")
                print("   - Ready for production deployment")
            elif overall_score >= 60:
                print("   ✅ GOOD: LLM integration working well") 
                print("   - Acceptable performance with room for optimization")
                print("   - Suitable for staging environments")
            else:
                print("   ⚠️ NEEDS IMPROVEMENT: LLM integration needs optimization")
                print("   - Performance or quality issues detected")
                print("   - Requires tuning before production use")

def main():
    """Run comprehensive LLM multiagent evaluation"""
    print("🤖 COMPREHENSIVE LLM MULTIAGENT EVALUATION")
    print("=" * 60)
    print("Evaluating LLM performance across all multiagent system components\n")
    
    evaluator = LLMMultiagentEvaluator()
    
    try:
        # Run all evaluation components
        evaluator.evaluate_llm_integration()
        evaluator.evaluate_agent_decision_making()
        evaluator.evaluate_fault_recovery_intelligence()
        evaluator.evaluate_negotiation_intelligence()
        
        # Generate comprehensive report
        evaluator.generate_evaluation_report()
        
        print(f"\n✅ LLM multiagent evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

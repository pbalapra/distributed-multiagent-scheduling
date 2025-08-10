#!/usr/bin/env python3
"""
LLM vs Heuristics Comparison Framework
=====================================

This script provides comprehensive comparison between LLM-enhanced 
decision making and traditional heuristic approaches in the multiagent system.
"""

import sys
import time
import json
import statistics
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.ollama_provider import OllamaProvider
from llm.llm_interface import LLMRequest, LLMManager, LLMProvider

class LLMHeuristicsComparator:
    """Compare LLM vs heuristic decision making"""
    
    def __init__(self):
        self.llm_provider = OllamaProvider(model_name="mistral", timeout=60.0)
        self.results = {
            "job_scoring": [],
            "fault_recovery": [], 
            "negotiation": [],
            "performance_metrics": {
                "llm_response_times": [],
                "heuristic_response_times": [],
                "llm_token_usage": []
            }
        }
    
    def compare_job_scoring(self):
        """Compare LLM vs heuristic job scoring decisions"""
        print("🎯 1. JOB SCORING: LLM vs HEURISTICS COMPARISON")
        print("=" * 60)
        
        # Test scenarios with different characteristics
        test_scenarios = [
            {
                "name": "Small High-Priority Job",
                "job": {"cpu_cores": 2, "memory_gb": 4, "priority": "HIGH", "runtime_minutes": 30},
                "resource": {"total_cpu": 16, "total_memory": 64, "current_utilization": 0.2}
            },
            {
                "name": "Large Medium-Priority Job", 
                "job": {"cpu_cores": 12, "memory_gb": 32, "priority": "MEDIUM", "runtime_minutes": 180},
                "resource": {"total_cpu": 16, "total_memory": 64, "current_utilization": 0.6}
            },
            {
                "name": "Critical Resource-Intensive Job",
                "job": {"cpu_cores": 16, "memory_gb": 48, "priority": "CRITICAL", "runtime_minutes": 240},
                "resource": {"total_cpu": 16, "total_memory": 64, "current_utilization": 0.8}
            },
            {
                "name": "Low-Priority Background Job",
                "job": {"cpu_cores": 1, "memory_gb": 2, "priority": "LOW", "runtime_minutes": 600},
                "resource": {"total_cpu": 16, "total_memory": 64, "current_utilization": 0.3}
            }
        ]
        
        print(f"\n📋 Testing {len(test_scenarios)} job scoring scenarios...\n")
        
        for scenario in test_scenarios:
            print(f"🔍 Scenario: {scenario['name']}")
            
            # Get LLM decision
            llm_score, llm_reasoning, llm_time = self._get_llm_job_score(scenario)
            
            # Get heuristic decision  
            heuristic_score, heuristic_reasoning, heuristic_time = self._get_heuristic_job_score(scenario)
            
            # Calculate difference
            score_diff = llm_score - heuristic_score
            time_diff = llm_time - heuristic_time
            
            # Store results
            self.results["job_scoring"].append({
                "scenario": scenario["name"],
                "llm_score": llm_score,
                "heuristic_score": heuristic_score,
                "score_difference": score_diff,
                "llm_time": llm_time,
                "heuristic_time": heuristic_time,
                "time_difference": time_diff,
                "llm_reasoning": llm_reasoning,
                "heuristic_reasoning": heuristic_reasoning
            })
            
            # Display comparison
            print(f"   📊 LLM Score: {llm_score:.3f} (reasoning: {llm_reasoning[:50]}...)")
            print(f"   🔢 Heuristic: {heuristic_score:.3f} (reasoning: {heuristic_reasoning})")
            print(f"   📈 Difference: {score_diff:+.3f} (LLM - Heuristic)")
            print(f"   ⏱️  Time: LLM={llm_time:.2f}s, Heuristic={heuristic_time:.3f}s (+{time_diff:.2f}s)")
            print()
    
    def compare_fault_recovery(self):
        """Compare LLM vs heuristic fault recovery strategies"""
        print("🛠️ 2. FAULT RECOVERY: LLM vs HEURISTICS COMPARISON")
        print("=" * 60)
        
        fault_scenarios = [
            {
                "name": "First Network Timeout",
                "failure": {
                    "type": "network_timeout",
                    "retry_count": 0,
                    "job_priority": "HIGH",
                    "deadline_minutes": 60
                },
                "context": {
                    "available_resources": ["worker-1", "worker-2"],
                    "system_load": 0.4
                }
            },
            {
                "name": "Repeated Resource Exhaustion",
                "failure": {
                    "type": "resource_exhaustion", 
                    "retry_count": 2,
                    "job_priority": "CRITICAL",
                    "deadline_minutes": 15
                },
                "context": {
                    "available_resources": ["worker-4"],
                    "system_load": 0.8
                }
            },
            {
                "name": "Hardware Failure with Limited Options",
                "failure": {
                    "type": "hardware_failure",
                    "retry_count": 3,
                    "job_priority": "MEDIUM", 
                    "deadline_minutes": 120
                },
                "context": {
                    "available_resources": [],
                    "system_load": 0.9
                }
            }
        ]
        
        print(f"\n🚨 Testing {len(fault_scenarios)} fault recovery scenarios...\n")
        
        for scenario in fault_scenarios:
            print(f"🔍 Scenario: {scenario['name']}")
            
            # Get LLM strategy
            llm_strategy, llm_reasoning, llm_time = self._get_llm_fault_recovery(scenario)
            
            # Get heuristic strategy
            heuristic_strategy, heuristic_reasoning, heuristic_time = self._get_heuristic_fault_recovery(scenario)
            
            # Store results
            self.results["fault_recovery"].append({
                "scenario": scenario["name"],
                "llm_strategy": llm_strategy,
                "heuristic_strategy": heuristic_strategy,
                "llm_time": llm_time,
                "heuristic_time": heuristic_time,
                "llm_reasoning": llm_reasoning,
                "heuristic_reasoning": heuristic_reasoning
            })
            
            # Display comparison
            print(f"   🤖 LLM Strategy: {llm_strategy} (reasoning: {llm_reasoning[:60]}...)")
            print(f"   🔢 Heuristic: {heuristic_strategy} (reasoning: {heuristic_reasoning})")
            print(f"   ⏱️  Time: LLM={llm_time:.2f}s, Heuristic={heuristic_time:.3f}s")
            print()
    
    def compare_negotiation(self):
        """Compare LLM vs heuristic negotiation decisions"""
        print("🤝 3. NEGOTIATION: LLM vs HEURISTICS COMPARISON") 
        print("=" * 60)
        
        negotiation_scenarios = [
            {
                "name": "Fair Resource Request (Low Load)",
                "proposal": {"cpu_cores": 4, "memory_gb": 8, "duration_minutes": 90, "credits": 120},
                "agent_state": {"available_cpu": 12, "available_memory": 48, "utilization": 0.3, "credits": 500}
            },
            {
                "name": "High Demand Request (Medium Load)",
                "proposal": {"cpu_cores": 8, "memory_gb": 16, "duration_minutes": 180, "credits": 200},
                "agent_state": {"available_cpu": 10, "available_memory": 20, "utilization": 0.7, "credits": 300}
            },
            {
                "name": "Oversubscribed Request (High Load)",
                "proposal": {"cpu_cores": 12, "memory_gb": 24, "duration_minutes": 120, "credits": 180},
                "agent_state": {"available_cpu": 4, "available_memory": 8, "utilization": 0.9, "credits": 100}
            }
        ]
        
        print(f"\n💼 Testing {len(negotiation_scenarios)} negotiation scenarios...\n")
        
        for scenario in negotiation_scenarios:
            print(f"🔍 Scenario: {scenario['name']}")
            
            # Get LLM decision
            llm_decision, llm_reasoning, llm_time = self._get_llm_negotiation(scenario)
            
            # Get heuristic decision
            heuristic_decision, heuristic_reasoning, heuristic_time = self._get_heuristic_negotiation(scenario)
            
            # Store results
            self.results["negotiation"].append({
                "scenario": scenario["name"],
                "llm_decision": llm_decision,
                "heuristic_decision": heuristic_decision,
                "llm_time": llm_time,
                "heuristic_time": heuristic_time,
                "llm_reasoning": llm_reasoning,
                "heuristic_reasoning": heuristic_reasoning
            })
            
            # Display comparison
            llm_action = "ACCEPT" if llm_decision else "REJECT"
            heuristic_action = "ACCEPT" if heuristic_decision else "REJECT"
            agreement = "✅ AGREE" if llm_decision == heuristic_decision else "❌ DISAGREE"
            
            print(f"   🤖 LLM: {llm_action} (reasoning: {llm_reasoning[:60]}...)")
            print(f"   🔢 Heuristic: {heuristic_action} (reasoning: {heuristic_reasoning})")
            print(f"   🤝 Agreement: {agreement}")
            print(f"   ⏱️  Time: LLM={llm_time:.2f}s, Heuristic={heuristic_time:.3f}s")
            print()
    
    def _get_llm_job_score(self, scenario) -> tuple:
        """Get LLM job scoring decision"""
        prompt = f"""You are an intelligent job scheduler. Analyze this scenario and provide a score (0.0-1.0).

JOB REQUIREMENTS:
- CPU Cores: {scenario['job']['cpu_cores']}
- Memory: {scenario['job']['memory_gb']} GB  
- Priority: {scenario['job']['priority']}
- Runtime: {scenario['job']['runtime_minutes']} minutes

RESOURCE STATUS:
- Total CPU: {scenario['resource']['total_cpu']} cores
- Total Memory: {scenario['resource']['total_memory']} GB
- Current Utilization: {scenario['resource']['current_utilization']:.1%}

Provide JSON: {{"score": <float>, "reasoning": "<explanation>"}}"""

        request = LLMRequest(prompt=prompt, context=scenario, task_type="job_scoring")
        
        start_time = time.time()
        response = self.llm_provider.generate_sync(request)
        llm_time = time.time() - start_time
        
        self.results["performance_metrics"]["llm_response_times"].append(llm_time)
        self.results["performance_metrics"]["llm_token_usage"].append(response.metadata.get("tokens_used", 0))
        
        try:
            data = json.loads(response.content)
            return data.get("score", 0.5), data.get("reasoning", "No reasoning provided"), llm_time
        except:
            return 0.5, "Invalid JSON response", llm_time
    
    def _get_heuristic_job_score(self, scenario) -> tuple:
        """Get heuristic job scoring decision"""
        start_time = time.time()
        
        job = scenario["job"]
        resource = scenario["resource"]
        
        # Simple heuristic scoring algorithm
        cpu_utilization = job["cpu_cores"] / resource["total_cpu"]
        memory_utilization = job["memory_gb"] / resource["total_memory"]
        current_load = resource["current_utilization"]
        
        # Priority weights
        priority_weights = {"LOW": 0.5, "MEDIUM": 0.7, "HIGH": 0.9, "CRITICAL": 1.0}
        priority_weight = priority_weights.get(job["priority"], 0.7)
        
        # Resource fit score (lower utilization is better)
        resource_fit = 1.0 - max(cpu_utilization, memory_utilization)
        
        # Load impact (lower current load is better)
        load_impact = 1.0 - current_load
        
        # Composite score
        score = (resource_fit * 0.5 + load_impact * 0.3 + priority_weight * 0.2)
        score = max(0.0, min(1.0, score))  # Clamp to [0,1]
        
        heuristic_time = time.time() - start_time
        self.results["performance_metrics"]["heuristic_response_times"].append(heuristic_time)
        
        reasoning = f"Resource fit: {resource_fit:.2f}, Load impact: {load_impact:.2f}, Priority: {priority_weight:.2f}"
        
        return score, reasoning, heuristic_time
    
    def _get_llm_fault_recovery(self, scenario) -> tuple:
        """Get LLM fault recovery strategy"""
        prompt = f"""You are a fault recovery system. Analyze this failure and recommend a strategy.

FAILURE CONTEXT:
- Type: {scenario['failure']['type']}
- Retry Count: {scenario['failure']['retry_count']}
- Job Priority: {scenario['failure']['job_priority']}
- Deadline: {scenario['failure']['deadline_minutes']} minutes

SYSTEM CONTEXT:
- Available Resources: {scenario['context']['available_resources']}
- System Load: {scenario['context']['system_load']:.1%}

Provide JSON: {{"strategy": "<strategy>", "reasoning": "<explanation>"}}"""

        request = LLMRequest(prompt=prompt, context=scenario, task_type="fault_recovery")
        
        start_time = time.time()
        response = self.llm_provider.generate_sync(request)
        llm_time = time.time() - start_time
        
        try:
            data = json.loads(response.content)
            return data.get("strategy", "retry"), data.get("reasoning", "No reasoning"), llm_time
        except:
            return "retry", "Invalid JSON response", llm_time
    
    def _get_heuristic_fault_recovery(self, scenario) -> tuple:
        """Get heuristic fault recovery strategy"""
        start_time = time.time()
        
        failure = scenario["failure"]
        context = scenario["context"]
        
        # Simple heuristic rules
        retry_count = failure["retry_count"]
        has_alternatives = len(context["available_resources"]) > 0
        is_critical = failure["job_priority"] == "CRITICAL"
        deadline_tight = failure["deadline_minutes"] < 30
        
        if retry_count >= 3:
            strategy = "escalate"
            reasoning = "Too many retries, escalating"
        elif not has_alternatives:
            strategy = "delay_retry" 
            reasoning = "No alternatives available, delaying"
        elif is_critical or deadline_tight:
            strategy = "immediate_retry"
            reasoning = "Critical priority or tight deadline"
        else:
            strategy = "retry_with_alternative"
            reasoning = "Standard retry with available alternative"
        
        heuristic_time = time.time() - start_time
        return strategy, reasoning, heuristic_time
    
    def _get_llm_negotiation(self, scenario) -> tuple:
        """Get LLM negotiation decision"""
        prompt = f"""You are a resource negotiation agent. Decide whether to accept this proposal.

PROPOSAL:
- CPU Cores: {scenario['proposal']['cpu_cores']}
- Memory: {scenario['proposal']['memory_gb']} GB
- Duration: {scenario['proposal']['duration_minutes']} minutes
- Credits Offered: {scenario['proposal']['credits']}

YOUR STATUS:
- Available CPU: {scenario['agent_state']['available_cpu']}
- Available Memory: {scenario['agent_state']['available_memory']} GB
- Current Utilization: {scenario['agent_state']['utilization']:.1%}
- Credit Balance: {scenario['agent_state']['credits']}

Provide JSON: {{"accept": <true/false>, "reasoning": "<explanation>"}}"""

        request = LLMRequest(prompt=prompt, context=scenario, task_type="negotiation")
        
        start_time = time.time()
        response = self.llm_provider.generate_sync(request)
        llm_time = time.time() - start_time
        
        try:
            data = json.loads(response.content)
            return data.get("accept", False), data.get("reasoning", "No reasoning"), llm_time
        except:
            return False, "Invalid JSON response", llm_time
    
    def _get_heuristic_negotiation(self, scenario) -> tuple:
        """Get heuristic negotiation decision"""
        start_time = time.time()
        
        proposal = scenario["proposal"]
        agent = scenario["agent_state"]
        
        # Simple heuristic rules
        cpu_fits = proposal["cpu_cores"] <= agent["available_cpu"]
        memory_fits = proposal["memory_gb"] <= agent["available_memory"]
        would_overload = agent["utilization"] > 0.8
        fair_credits = proposal["credits"] > 100  # Minimum threshold
        
        accept = cpu_fits and memory_fits and not would_overload and fair_credits
        
        if not cpu_fits or not memory_fits:
            reasoning = "Insufficient resources available"
        elif would_overload:
            reasoning = "Would exceed utilization threshold"
        elif not fair_credits:
            reasoning = "Credits offered too low"
        else:
            reasoning = "Proposal meets all criteria"
        
        heuristic_time = time.time() - start_time
        return accept, reasoning, heuristic_time
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n📊 COMPREHENSIVE LLM vs HEURISTICS COMPARISON REPORT")
        print("=" * 80)
        
        # Performance Analysis
        llm_times = self.results["performance_metrics"]["llm_response_times"]
        heuristic_times = self.results["performance_metrics"]["heuristic_response_times"]
        
        if llm_times and heuristic_times:
            avg_llm_time = statistics.mean(llm_times)
            avg_heuristic_time = statistics.mean(heuristic_times)
            speed_ratio = avg_llm_time / avg_heuristic_time
            
            print(f"\n⏱️ PERFORMANCE ANALYSIS:")
            print(f"   LLM Average Response Time: {avg_llm_time:.2f}s")
            print(f"   Heuristic Average Time: {avg_heuristic_time:.4f}s") 
            print(f"   Speed Ratio: LLM is {speed_ratio:.0f}x slower than heuristics")
            print(f"   LLM Token Usage: {statistics.mean(self.results['performance_metrics']['llm_token_usage']):.0f} tokens/request")
        
        # Job Scoring Analysis
        if self.results["job_scoring"]:
            print(f"\n🎯 JOB SCORING ANALYSIS:")
            job_improvements = [r["score_difference"] for r in self.results["job_scoring"]]
            avg_improvement = statistics.mean(job_improvements)
            positive_improvements = sum(1 for diff in job_improvements if diff > 0)
            
            print(f"   Average Score Difference: {avg_improvement:+.3f} (LLM - Heuristic)")
            print(f"   LLM Better Decisions: {positive_improvements}/{len(job_improvements)}")
            
            print(f"\n   📋 Detailed Job Scoring Results:")
            for result in self.results["job_scoring"]:
                status = "🟢" if result["score_difference"] > 0 else "🔴" if result["score_difference"] < 0 else "🟡"
                print(f"   {status} {result['scenario']}: {result['score_difference']:+.3f}")
        
        # Fault Recovery Analysis
        if self.results["fault_recovery"]:
            print(f"\n🛠️ FAULT RECOVERY ANALYSIS:")
            print(f"   Strategy Comparisons:")
            
            agreement_count = 0
            for result in self.results["fault_recovery"]:
                agrees = result["llm_strategy"] == result["heuristic_strategy"]
                if agrees:
                    agreement_count += 1
                status = "✅" if agrees else "❌"
                print(f"   {status} {result['scenario']}: LLM={result['llm_strategy']}, Heuristic={result['heuristic_strategy']}")
            
            agreement_rate = (agreement_count / len(self.results["fault_recovery"])) * 100
            print(f"   Agreement Rate: {agreement_rate:.1f}%")
        
        # Negotiation Analysis
        if self.results["negotiation"]:
            print(f"\n🤝 NEGOTIATION ANALYSIS:")
            print(f"   Decision Comparisons:")
            
            agreement_count = 0
            for result in self.results["negotiation"]:
                agrees = result["llm_decision"] == result["heuristic_decision"]
                if agrees:
                    agreement_count += 1
                status = "✅" if agrees else "❌"
                llm_action = "ACCEPT" if result["llm_decision"] else "REJECT"
                heuristic_action = "ACCEPT" if result["heuristic_decision"] else "REJECT"
                print(f"   {status} {result['scenario']}: LLM={llm_action}, Heuristic={heuristic_action}")
            
            agreement_rate = (agreement_count / len(self.results["negotiation"])) * 100
            print(f"   Agreement Rate: {agreement_rate:.1f}%")
        
        # Overall Assessment
        print(f"\n🎯 OVERALL COMPARISON ASSESSMENT:")
        
        # Calculate overall scores
        decision_quality_score = 50  # Base score
        
        if self.results["job_scoring"]:
            # Bonus for positive job scoring improvements
            avg_improvement = statistics.mean([r["score_difference"] for r in self.results["job_scoring"]])
            decision_quality_score += max(-30, min(30, avg_improvement * 100))
        
        performance_score = 100
        if llm_times:
            # Penalty for slow responses (target: under 10s)
            avg_time = statistics.mean(llm_times)
            if avg_time > 10:
                performance_score = max(0, 100 - (avg_time - 10) * 5)
        
        overall_score = (decision_quality_score + performance_score) / 2
        
        print(f"   Decision Quality Score: {decision_quality_score:.1f}/100")
        print(f"   Performance Score: {performance_score:.1f}/100")
        print(f"   Overall Score: {overall_score:.1f}/100")
        
        if overall_score >= 80:
            print(f"\n   🎉 EXCELLENT: LLM provides superior intelligent decision making")
            print(f"   - Worth the performance cost for better decisions")
            print(f"   - Recommended for production use")
        elif overall_score >= 60:
            print(f"\n   ✅ GOOD: LLM provides value in most scenarios")
            print(f"   - Mixed results with some clear improvements")
            print(f"   - Consider selective LLM use for complex decisions")
        else:
            print(f"\n   ⚠️ QUESTIONABLE: Heuristics may be sufficient")
            print(f"   - Limited improvement over heuristics")
            print(f"   - Consider optimizing LLM integration")

def main():
    """Run comprehensive LLM vs heuristics comparison"""
    print("🔬 LLM vs HEURISTICS COMPREHENSIVE COMPARISON")
    print("=" * 60)
    print("Comparing intelligent LLM decision making against traditional heuristics\n")
    
    comparator = LLMHeuristicsComparator()
    
    try:
        comparator.compare_job_scoring()
        comparator.compare_fault_recovery()
        comparator.compare_negotiation()
        comparator.generate_comparison_report()
        
        print(f"\n✅ Comparison analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

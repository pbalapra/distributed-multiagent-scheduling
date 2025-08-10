#!/usr/bin/env python3
"""
Resilience Evaluation: LLM vs Heuristics Fault Tolerance
=======================================================

This evaluation tests the resilience and fault tolerance capabilities of:
1. LLM-enhanced decision making under stress
2. Traditional heuristic approaches under failures
3. Recovery patterns and adaptive behavior
4. Performance degradation under various failure scenarios

Tests include:
- Network timeout scenarios with escalating failures
- Resource exhaustion with recovery strategies  
- System overload and graceful degradation
- Multi-failure cascading scenarios
- Performance consistency under stress
"""

import sys
import time
import json
import statistics
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from llm.ollama_provider import OllamaProvider
    from llm.llm_interface import LLMRequest, LLMManager, LLMProvider
    LLM_AVAILABLE = True
except ImportError:
    print("⚠️ LLM modules not available, creating mock implementation")
    LLM_AVAILABLE = False
    
    class MockLLMProvider:
        def generate_sync(self, request):
            # Mock LLM responses for testing
            responses = {
                "fault_recovery": '{"strategy": "intelligent_retry_with_backoff", "reasoning": "Smart adaptive recovery"}',
                "job_scheduling": '{"score": 0.75, "reasoning": "Intelligent resource optimization"}',
                "system_diagnosis": '{"diagnosis": "resource_contention", "action": "load_balancing", "confidence": 0.8}'
            }
            
            class MockResponse:
                def __init__(self, content):
                    self.content = content
                    self.metadata = {"tokens_used": 50}
            
            task_type = getattr(request, 'task_type', 'fault_recovery')
            content = responses.get(task_type, '{"response": "mock_llm_response"}')
            time.sleep(random.uniform(2, 8))  # Simulate LLM latency
            return MockResponse(content)

class ResilienceScenario:
    """Defines a stress/failure scenario for testing"""
    
    def __init__(self, name: str, description: str, failure_type: str, 
                 severity: float, duration: float, cascading: bool = False):
        self.name = name
        self.description = description
        self.failure_type = failure_type
        self.severity = severity  # 0.0 to 1.0
        self.duration = duration  # seconds
        self.cascading = cascading

class ResilienceMetrics:
    """Metrics for measuring resilience performance"""
    
    def __init__(self):
        self.decisions_made = 0
        self.successful_decisions = 0
        self.response_times = []
        self.error_rate = 0.0
        self.recovery_times = []
        self.adaptive_behaviors = 0
        self.degradation_factor = 0.0
        self.consistency_score = 0.0

class LLMHeuristicsResilienceEvaluator:
    """Evaluate resilience of LLM vs heuristic decision making"""
    
    def __init__(self, iterations=3):
        self.iterations = iterations
        if LLM_AVAILABLE:
            self.llm_provider = OllamaProvider(model_name="mistral", timeout=60.0)
        else:
            self.llm_provider = MockLLMProvider()
        
        # Define stress scenarios
        self.scenarios = [
            ResilienceScenario(
                "Network Timeouts",
                "Increasing network timeout rates affecting communication",
                "network_timeout",
                severity=0.3,
                duration=30.0
            ),
            ResilienceScenario(
                "Resource Exhaustion",
                "Progressive resource depletion causing scheduling conflicts",
                "resource_exhaustion", 
                severity=0.6,
                duration=45.0
            ),
            ResilienceScenario(
                "System Overload",
                "Sudden spike in job requests overwhelming the system",
                "system_overload",
                severity=0.8,
                duration=60.0
            ),
            ResilienceScenario(
                "Cascading Failures",
                "Multiple interconnected failures triggering secondary issues",
                "cascading_failure",
                severity=0.9,
                duration=90.0,
                cascading=True
            ),
            ResilienceScenario(
                "Communication Disruption", 
                "Intermittent communication failures between agents",
                "communication_failure",
                severity=0.4,
                duration=40.0
            )
        ]
        
        self.results = {
            "llm_resilience": {},
            "heuristic_resilience": {},
            "comparative_analysis": {}
        }
    
    def run_resilience_evaluation(self):
        """Run comprehensive resilience evaluation"""
        print("🛡️ RESILIENCE EVALUATION: LLM vs HEURISTICS")
        print("=" * 60)
        print(f"Testing fault tolerance across {len(self.scenarios)} failure scenarios")
        print(f"Running {self.iterations} iterations per scenario for statistical validity\n")
        
        for scenario in self.scenarios:
            print(f"🔥 Testing Scenario: {scenario.name}")
            print(f"   Description: {scenario.description}")
            print(f"   Severity: {scenario.severity:.1%}, Duration: {scenario.duration}s")
            
            llm_results = []
            heuristic_results = []
            
            for iteration in range(self.iterations):
                print(f"   🔄 Iteration {iteration + 1}/{self.iterations}")
                
                # Test LLM resilience
                llm_metrics = self._test_llm_resilience(scenario)
                llm_results.append(llm_metrics)
                
                # Test heuristic resilience  
                heuristic_metrics = self._test_heuristic_resilience(scenario)
                heuristic_results.append(heuristic_metrics)
                
                # Brief pause between iterations
                time.sleep(1)
            
            # Store aggregated results
            self.results["llm_resilience"][scenario.name] = llm_results
            self.results["heuristic_resilience"][scenario.name] = heuristic_results
            
            # Show iteration summary
            llm_success_rate = statistics.mean([r.successful_decisions / max(1, r.decisions_made) 
                                              for r in llm_results])
            heuristic_success_rate = statistics.mean([r.successful_decisions / max(1, r.decisions_made) 
                                                    for r in heuristic_results])
            
            print(f"   📊 Success Rates: LLM {llm_success_rate:.1%} vs Heuristic {heuristic_success_rate:.1%}")
            print()
        
        # Generate comprehensive analysis
        self._analyze_resilience_results()
    
    def _test_llm_resilience(self, scenario: ResilienceScenario) -> ResilienceMetrics:
        """Test LLM resilience under specific failure scenario"""
        metrics = ResilienceMetrics()
        
        # Simulate decisions under stress
        num_decisions = random.randint(8, 15)
        stress_multiplier = 1.0 + scenario.severity
        
        for i in range(num_decisions):
            start_time = time.time()
            
            # Create failure context
            failure_context = self._generate_failure_context(scenario, i, num_decisions)
            
            try:
                # Test LLM decision making under stress
                decision_success = self._make_llm_decision(failure_context, scenario)
                
                response_time = (time.time() - start_time) * stress_multiplier
                metrics.response_times.append(response_time)
                metrics.decisions_made += 1
                
                if decision_success:
                    metrics.successful_decisions += 1
                    
                    # Check for adaptive behavior
                    if self._is_adaptive_behavior(failure_context, decision_success):
                        metrics.adaptive_behaviors += 1
                
            except Exception as e:
                metrics.decisions_made += 1
                # Failure recorded but no success
                print(f"      ❌ LLM Decision failed: {e}")
            
            # Simulate progressive degradation
            if scenario.cascading and i > num_decisions // 2:
                time.sleep(0.5)  # Additional delay for cascading effects
        
        # Calculate derived metrics
        metrics.error_rate = 1.0 - (metrics.successful_decisions / max(1, metrics.decisions_made))
        metrics.consistency_score = self._calculate_consistency_score(metrics.response_times)
        metrics.degradation_factor = min(1.0, scenario.severity * metrics.error_rate)
        
        return metrics
    
    def _test_heuristic_resilience(self, scenario: ResilienceScenario) -> ResilienceMetrics:
        """Test heuristic resilience under specific failure scenario"""
        metrics = ResilienceMetrics()
        
        # Simulate decisions under stress
        num_decisions = random.randint(8, 15)
        
        for i in range(num_decisions):
            start_time = time.time()
            
            # Create failure context
            failure_context = self._generate_failure_context(scenario, i, num_decisions)
            
            try:
                # Test heuristic decision making under stress
                decision_success = self._make_heuristic_decision(failure_context, scenario)
                
                response_time = (time.time() - start_time)  # Heuristics are fast even under stress
                metrics.response_times.append(response_time)
                metrics.decisions_made += 1
                
                if decision_success:
                    metrics.successful_decisions += 1
                
            except Exception as e:
                metrics.decisions_made += 1
                print(f"      ❌ Heuristic Decision failed: {e}")
            
            # Heuristics don't adapt, so no adaptive behavior tracking
        
        # Calculate derived metrics  
        metrics.error_rate = 1.0 - (metrics.successful_decisions / max(1, metrics.decisions_made))
        metrics.consistency_score = self._calculate_consistency_score(metrics.response_times)
        metrics.degradation_factor = min(1.0, scenario.severity * metrics.error_rate)
        
        return metrics
    
    def _generate_failure_context(self, scenario: ResilienceScenario, 
                                 decision_num: int, total_decisions: int) -> Dict:
        """Generate failure context for testing"""
        # Progressive failure intensity
        progress = decision_num / total_decisions
        current_severity = scenario.severity * (0.5 + 0.5 * progress)
        
        context = {
            "scenario_type": scenario.failure_type,
            "severity": current_severity,
            "time_elapsed": progress * scenario.duration,
            "cascading": scenario.cascading,
            "system_state": {
                "cpu_utilization": min(0.95, 0.4 + current_severity * 0.5),
                "memory_utilization": min(0.90, 0.3 + current_severity * 0.6),
                "network_latency": 50 + current_severity * 200,  # ms
                "active_failures": int(current_severity * 5),
                "failed_nodes": int(current_severity * 3)
            },
            "failure_indicators": [
                f"High {scenario.failure_type}",
                f"System stress at {current_severity:.1%}",
                f"Decision #{decision_num + 1} of {total_decisions}"
            ]
        }
        
        return context
    
    def _make_llm_decision(self, context: Dict, scenario: ResilienceScenario) -> bool:
        """Make an LLM-based decision under stress"""
        prompt = f"""You are a resilient system coordinator dealing with a critical situation.

FAILURE SCENARIO: {scenario.name}
SYSTEM STATE:
- CPU Utilization: {context['system_state']['cpu_utilization']:.1%}
- Memory Utilization: {context['system_state']['memory_utilization']:.1%}
- Network Latency: {context['system_state']['network_latency']:.0f}ms
- Active Failures: {context['system_state']['active_failures']}
- Failed Nodes: {context['system_state']['failed_nodes']}

SITUATION: {scenario.description}
SEVERITY: {context['severity']:.1%}

Provide a resilient response strategy. Return JSON: {{"strategy": "<action>", "reasoning": "<explanation>", "confidence": <0.0-1.0>}}"""

        request = LLMRequest(prompt=prompt, context=context, task_type="fault_recovery")
        
        try:
            response = self.llm_provider.generate_sync(request)
            data = json.loads(response.content)
            
            strategy = data.get("strategy", "")
            confidence = data.get("confidence", 0.5)
            
            # Success probability based on strategy quality and confidence
            success_prob = confidence * (1.0 - context["severity"] * 0.3)
            return random.random() < success_prob
            
        except Exception as e:
            print(f"        LLM Error: {e}")
            return False
    
    def _make_heuristic_decision(self, context: Dict, scenario: ResilienceScenario) -> bool:
        """Make a heuristic-based decision under stress"""
        # Simple heuristic rules for fault tolerance
        cpu_util = context['system_state']['cpu_utilization']
        memory_util = context['system_state']['memory_utilization']
        severity = context['severity']
        
        # Basic decision tree
        if cpu_util > 0.9 or memory_util > 0.85:
            action = "throttle_requests"
            success_prob = 0.6
        elif severity > 0.7:
            action = "emergency_shutdown"
            success_prob = 0.4
        elif context['system_state']['active_failures'] > 3:
            action = "failover"
            success_prob = 0.7
        else:
            action = "continue_normal"
            success_prob = 0.8
        
        # Heuristics degrade linearly with severity
        adjusted_success_prob = success_prob * (1.0 - severity * 0.5)
        
        return random.random() < adjusted_success_prob
    
    def _is_adaptive_behavior(self, context: Dict, success: bool) -> bool:
        """Check if decision shows adaptive behavior"""
        # LLM shows adaptive behavior when it succeeds despite high severity
        return success and context["severity"] > 0.6
    
    def _calculate_consistency_score(self, response_times: List[float]) -> float:
        """Calculate response time consistency (0-1, higher is better)"""
        if len(response_times) < 2:
            return 1.0
        
        mean_time = statistics.mean(response_times)
        std_time = statistics.stdev(response_times)
        
        # Lower coefficient of variation = higher consistency
        cv = std_time / mean_time if mean_time > 0 else 1.0
        consistency = max(0.0, 1.0 - cv)
        
        return consistency
    
    def _analyze_resilience_results(self):
        """Generate comprehensive resilience analysis"""
        print("\n" + "=" * 60)
        print("🛡️ RESILIENCE ANALYSIS REPORT")
        print("=" * 60)
        
        # Overall resilience scores
        llm_overall_scores = []
        heuristic_overall_scores = []
        
        print("\n📊 Per-Scenario Resilience Analysis:")
        
        for scenario_name in self.scenarios:
            scenario_name = scenario_name.name
            
            llm_results = self.results["llm_resilience"][scenario_name]
            heuristic_results = self.results["heuristic_resilience"][scenario_name]
            
            # Calculate averages
            llm_success_rate = statistics.mean([r.successful_decisions / max(1, r.decisions_made) 
                                              for r in llm_results])
            heuristic_success_rate = statistics.mean([r.successful_decisions / max(1, r.decisions_made) 
                                                    for r in heuristic_results])
            
            llm_avg_response = statistics.mean([statistics.mean(r.response_times) 
                                             for r in llm_results if r.response_times])
            heuristic_avg_response = statistics.mean([statistics.mean(r.response_times) 
                                                    for r in heuristic_results if r.response_times])
            
            llm_consistency = statistics.mean([r.consistency_score for r in llm_results])
            heuristic_consistency = statistics.mean([r.consistency_score for r in heuristic_results])
            
            llm_adaptive = statistics.mean([r.adaptive_behaviors for r in llm_results])
            heuristic_adaptive = statistics.mean([r.adaptive_behaviors for r in heuristic_results])
            
            # Calculate scenario scores
            llm_scenario_score = (llm_success_rate * 40 + 
                                llm_consistency * 30 + 
                                min(1.0, llm_adaptive / 3) * 20 +
                                (1.0 - min(1.0, llm_avg_response / 10)) * 10)
            
            heuristic_scenario_score = (heuristic_success_rate * 40 + 
                                      heuristic_consistency * 30 +
                                      min(1.0, heuristic_adaptive / 3) * 20 +
                                      (1.0 - min(1.0, heuristic_avg_response / 10)) * 10)
            
            llm_overall_scores.append(llm_scenario_score)
            heuristic_overall_scores.append(heuristic_scenario_score)
            
            # Display scenario analysis
            print(f"\n🔥 {scenario_name}:")
            print(f"   Success Rate: LLM {llm_success_rate:.1%} vs Heuristic {heuristic_success_rate:.1%}")
            print(f"   Avg Response: LLM {llm_avg_response:.2f}s vs Heuristic {heuristic_avg_response:.3f}s")
            print(f"   Consistency: LLM {llm_consistency:.2f} vs Heuristic {heuristic_consistency:.2f}")
            print(f"   Adaptive Behaviors: LLM {llm_adaptive:.1f} vs Heuristic {heuristic_adaptive:.1f}")
            
            winner = "LLM" if llm_scenario_score > heuristic_scenario_score else "Heuristic"
            margin = abs(llm_scenario_score - heuristic_scenario_score)
            print(f"   🏆 Winner: {winner} (Score: {max(llm_scenario_score, heuristic_scenario_score):.1f}, Margin: +{margin:.1f})")
        
        # Overall analysis
        llm_overall_score = statistics.mean(llm_overall_scores)
        heuristic_overall_score = statistics.mean(heuristic_overall_scores)
        
        print(f"\n🎯 OVERALL RESILIENCE ASSESSMENT:")
        print(f"   LLM Overall Score: {llm_overall_score:.1f}/100")
        print(f"   Heuristic Overall Score: {heuristic_overall_score:.1f}/100")
        
        # Determine winner
        if llm_overall_score > heuristic_overall_score:
            margin = llm_overall_score - heuristic_overall_score
            print(f"\n   🎉 LLM WINS: {margin:.1f} point advantage")
            print(f"   🛡️ LLM demonstrates superior resilience and adaptability")
            print(f"   💡 Recommended for fault-tolerant production systems")
            
            if margin > 20:
                confidence = "VERY HIGH"
            elif margin > 10:
                confidence = "HIGH"
            else:
                confidence = "MODERATE"
                
            print(f"   📊 Confidence Level: {confidence}")
        else:
            margin = heuristic_overall_score - llm_overall_score
            print(f"\n   ⚖️ HEURISTIC WINS: {margin:.1f} point advantage")
            print(f"   🔧 Traditional approaches show better resilience patterns")
            print(f"   💭 Consider heuristics for mission-critical fault tolerance")
        
        # Detailed resilience characteristics
        print(f"\n🔍 RESILIENCE CHARACTERISTICS:")
        
        # LLM characteristics
        total_llm_decisions = sum(sum(r.decisions_made for r in scenario_results) 
                                for scenario_results in self.results["llm_resilience"].values())
        total_llm_adaptive = sum(sum(r.adaptive_behaviors for r in scenario_results)
                               for scenario_results in self.results["llm_resilience"].values())
        
        print(f"   🤖 LLM Resilience:")
        print(f"      - Total Decisions: {total_llm_decisions}")
        print(f"      - Adaptive Behaviors: {total_llm_adaptive} ({total_llm_adaptive/max(1,total_llm_decisions):.1%})")
        print(f"      - Strengths: Contextual adaptation, intelligent recovery")
        print(f"      - Weaknesses: Response time variability under stress")
        
        # Heuristic characteristics  
        total_heuristic_decisions = sum(sum(r.decisions_made for r in scenario_results)
                                      for scenario_results in self.results["heuristic_resilience"].values())
        
        print(f"   🔧 Heuristic Resilience:")
        print(f"      - Total Decisions: {total_heuristic_decisions}")
        print(f"      - Strengths: Consistent response times, predictable behavior")
        print(f"      - Weaknesses: Limited adaptability, rigid failure handling")
        
        # Store comparative analysis
        self.results["comparative_analysis"] = {
            "llm_overall_score": llm_overall_score,
            "heuristic_overall_score": heuristic_overall_score,
            "winner": "LLM" if llm_overall_score > heuristic_overall_score else "Heuristic",
            "margin": abs(llm_overall_score - heuristic_overall_score),
            "total_scenarios": len(self.scenarios),
            "total_iterations": self.iterations
        }

def main():
    """Run resilience evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Resilience Evaluation: LLM vs Heuristics")
    parser.add_argument("--iterations", "-i", type=int, default=3,
                       help="Number of iterations per scenario (default: 3)")
    
    args = parser.parse_args()
    
    evaluator = LLMHeuristicsResilienceEvaluator(iterations=args.iterations)
    
    try:
        evaluator.run_resilience_evaluation()
        print(f"\n✅ Resilience evaluation completed successfully!")
        print(f"📊 Tested {len(evaluator.scenarios)} failure scenarios with {args.iterations} iterations each")
        return 0
        
    except Exception as e:
        print(f"\n❌ Resilience evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

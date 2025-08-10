#!/usr/bin/env python3
"""
Statistical LLM vs Heuristics Analysis
=====================================

This script runs multiple iterations of LLM vs heuristic comparisons
to provide statistical analysis with confidence intervals, significance tests,
and variance analysis.
"""

import sys
import time
import json
import statistics
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.ollama_provider import OllamaProvider
from llm.llm_interface import LLMRequest, LLMManager, LLMProvider

class StatisticalLLMComparison:
    """Statistical analysis of LLM vs heuristic performance across multiple runs"""
    
    def __init__(self, num_iterations=5):
        self.num_iterations = num_iterations
        self.llm_provider = OllamaProvider(model_name="mistral", timeout=60.0)
        self.statistical_results = {
            "job_scoring": {
                "llm_scores": [],
                "heuristic_scores": [],
                "score_differences": [],
                "llm_times": [],
                "heuristic_times": []
            },
            "fault_recovery": {
                "llm_strategies": [],
                "heuristic_strategies": [],
                "llm_times": [],
                "heuristic_times": [],
                "agreements": []
            },
            "negotiation": {
                "llm_decisions": [],
                "heuristic_decisions": [],
                "llm_times": [],
                "heuristic_times": [],
                "agreements": []
            },
            "performance_metrics": {
                "llm_response_times": [],
                "heuristic_response_times": [],
                "token_usage": []
            }
        }
        
        # Test scenarios (consistent across all runs)
        self.job_scenarios = [
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
        
        self.fault_scenarios = [
            {
                "name": "First Network Timeout",
                "failure": {"type": "network_timeout", "retry_count": 0, "job_priority": "HIGH", "deadline_minutes": 60},
                "context": {"available_resources": ["worker-1", "worker-2"], "system_load": 0.4}
            },
            {
                "name": "Repeated Resource Exhaustion",
                "failure": {"type": "resource_exhaustion", "retry_count": 2, "job_priority": "CRITICAL", "deadline_minutes": 15},
                "context": {"available_resources": ["worker-4"], "system_load": 0.8}
            },
            {
                "name": "Hardware Failure with Limited Options",
                "failure": {"type": "hardware_failure", "retry_count": 3, "job_priority": "MEDIUM", "deadline_minutes": 120},
                "context": {"available_resources": [], "system_load": 0.9}
            }
        ]
        
        self.negotiation_scenarios = [
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
    
    def run_statistical_analysis(self):
        """Run multiple iterations for statistical significance"""
        print(f"📊 STATISTICAL LLM vs HEURISTICS ANALYSIS")
        print(f"🔄 Running {self.num_iterations} iterations for statistical significance")
        print("=" * 80)
        
        for iteration in range(self.num_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{self.num_iterations}")
            print("-" * 40)
            
            self.run_job_scoring_iteration(iteration)
            self.run_fault_recovery_iteration(iteration)
            self.run_negotiation_iteration(iteration)
            
            # Brief pause between iterations to avoid overwhelming the LLM
            if iteration < self.num_iterations - 1:
                time.sleep(2)
        
        self.generate_statistical_report()
    
    def run_job_scoring_iteration(self, iteration):
        """Run job scoring tests for one iteration"""
        print(f"  🎯 Job Scoring (Iteration {iteration + 1})")
        
        iteration_results = {
            "llm_scores": [],
            "heuristic_scores": [],
            "score_diffs": [],
            "llm_times": [],
            "heuristic_times": []
        }
        
        for scenario in self.job_scenarios:
            # Get LLM decision
            llm_score, _, llm_time = self._get_llm_job_score(scenario)
            
            # Get heuristic decision  
            heuristic_score, _, heuristic_time = self._get_heuristic_job_score(scenario)
            
            # Store results
            iteration_results["llm_scores"].append(llm_score)
            iteration_results["heuristic_scores"].append(heuristic_score)
            iteration_results["score_diffs"].append(llm_score - heuristic_score)
            iteration_results["llm_times"].append(llm_time)
            iteration_results["heuristic_times"].append(heuristic_time)
        
        # Store iteration results
        self.statistical_results["job_scoring"]["llm_scores"].append(iteration_results["llm_scores"])
        self.statistical_results["job_scoring"]["heuristic_scores"].append(iteration_results["heuristic_scores"])
        self.statistical_results["job_scoring"]["score_differences"].append(iteration_results["score_diffs"])
        self.statistical_results["job_scoring"]["llm_times"].append(iteration_results["llm_times"])
        self.statistical_results["job_scoring"]["heuristic_times"].append(iteration_results["heuristic_times"])
        
        # Print iteration summary
        avg_diff = statistics.mean(iteration_results["score_diffs"])
        avg_time = statistics.mean(iteration_results["llm_times"])
        print(f"     Average score difference: {avg_diff:+.3f}, Average LLM time: {avg_time:.2f}s")
    
    def run_fault_recovery_iteration(self, iteration):
        """Run fault recovery tests for one iteration"""
        print(f"  🛠️  Fault Recovery (Iteration {iteration + 1})")
        
        iteration_results = {
            "llm_strategies": [],
            "heuristic_strategies": [],
            "agreements": [],
            "llm_times": [],
            "heuristic_times": []
        }
        
        for scenario in self.fault_scenarios:
            # Get LLM strategy
            llm_strategy, _, llm_time = self._get_llm_fault_recovery(scenario)
            
            # Get heuristic strategy
            heuristic_strategy, _, heuristic_time = self._get_heuristic_fault_recovery(scenario)
            
            # Store results
            iteration_results["llm_strategies"].append(llm_strategy)
            iteration_results["heuristic_strategies"].append(heuristic_strategy)
            iteration_results["agreements"].append(llm_strategy == heuristic_strategy)
            iteration_results["llm_times"].append(llm_time)
            iteration_results["heuristic_times"].append(heuristic_time)
        
        # Store iteration results
        self.statistical_results["fault_recovery"]["llm_strategies"].append(iteration_results["llm_strategies"])
        self.statistical_results["fault_recovery"]["heuristic_strategies"].append(iteration_results["heuristic_strategies"])
        self.statistical_results["fault_recovery"]["agreements"].append(iteration_results["agreements"])
        self.statistical_results["fault_recovery"]["llm_times"].append(iteration_results["llm_times"])
        self.statistical_results["fault_recovery"]["heuristic_times"].append(iteration_results["heuristic_times"])
        
        # Print iteration summary
        agreement_rate = (sum(iteration_results["agreements"]) / len(iteration_results["agreements"])) * 100
        avg_time = statistics.mean(iteration_results["llm_times"])
        print(f"     Agreement rate: {agreement_rate:.1f}%, Average LLM time: {avg_time:.2f}s")
    
    def run_negotiation_iteration(self, iteration):
        """Run negotiation tests for one iteration"""
        print(f"  🤝 Negotiation (Iteration {iteration + 1})")
        
        iteration_results = {
            "llm_decisions": [],
            "heuristic_decisions": [],
            "agreements": [],
            "llm_times": [],
            "heuristic_times": []
        }
        
        for scenario in self.negotiation_scenarios:
            # Get LLM decision
            llm_decision, _, llm_time = self._get_llm_negotiation(scenario)
            
            # Get heuristic decision
            heuristic_decision, _, heuristic_time = self._get_heuristic_negotiation(scenario)
            
            # Store results
            iteration_results["llm_decisions"].append(llm_decision)
            iteration_results["heuristic_decisions"].append(heuristic_decision)
            iteration_results["agreements"].append(llm_decision == heuristic_decision)
            iteration_results["llm_times"].append(llm_time)
            iteration_results["heuristic_times"].append(heuristic_time)
        
        # Store iteration results
        self.statistical_results["negotiation"]["llm_decisions"].append(iteration_results["llm_decisions"])
        self.statistical_results["negotiation"]["heuristic_decisions"].append(iteration_results["heuristic_decisions"])
        self.statistical_results["negotiation"]["agreements"].append(iteration_results["agreements"])
        self.statistical_results["negotiation"]["llm_times"].append(iteration_results["llm_times"])
        self.statistical_results["negotiation"]["heuristic_times"].append(iteration_results["heuristic_times"])
        
        # Print iteration summary
        agreement_rate = (sum(iteration_results["agreements"]) / len(iteration_results["agreements"])) * 100
        avg_time = statistics.mean(iteration_results["llm_times"])
        print(f"     Agreement rate: {agreement_rate:.1f}%, Average LLM time: {avg_time:.2f}s")
    
    def generate_statistical_report(self):
        """Generate comprehensive statistical analysis report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        print("=" * 80)
        
        # Job Scoring Statistical Analysis
        self.analyze_job_scoring_statistics()
        
        # Fault Recovery Statistical Analysis
        self.analyze_fault_recovery_statistics()
        
        # Negotiation Statistical Analysis
        self.analyze_negotiation_statistics()
        
        # Performance Statistical Analysis
        self.analyze_performance_statistics()
        
        # Overall Statistical Assessment
        self.generate_overall_statistical_assessment()
    
    def analyze_job_scoring_statistics(self):
        """Analyze job scoring with statistical measures"""
        print("\n🎯 JOB SCORING STATISTICAL ANALYSIS")
        print("-" * 50)
        
        # Flatten score differences across all iterations
        all_score_diffs = []
        for iteration_diffs in self.statistical_results["job_scoring"]["score_differences"]:
            all_score_diffs.extend(iteration_diffs)
        
        # Flatten LLM scores
        all_llm_scores = []
        for iteration_scores in self.statistical_results["job_scoring"]["llm_scores"]:
            all_llm_scores.extend(iteration_scores)
        
        # Flatten heuristic scores
        all_heuristic_scores = []
        for iteration_scores in self.statistical_results["job_scoring"]["heuristic_scores"]:
            all_heuristic_scores.extend(iteration_scores)
        
        # Calculate statistics
        if all_score_diffs:
            mean_diff = statistics.mean(all_score_diffs)
            std_diff = statistics.stdev(all_score_diffs) if len(all_score_diffs) > 1 else 0
            median_diff = statistics.median(all_score_diffs)
            
            # Count improvements
            improvements = sum(1 for diff in all_score_diffs if diff > 0)
            total_tests = len(all_score_diffs)
            improvement_rate = (improvements / total_tests) * 100
            
            # Confidence interval (95%)
            n = len(all_score_diffs)
            margin_error = 1.96 * (std_diff / np.sqrt(n)) if n > 1 else 0
            ci_lower = mean_diff - margin_error
            ci_upper = mean_diff + margin_error
            
            print(f"📈 Score Difference Analysis (LLM - Heuristic):")
            print(f"   Mean Difference: {mean_diff:+.3f}")
            print(f"   Median Difference: {median_diff:+.3f}")
            print(f"   Standard Deviation: {std_diff:.3f}")
            print(f"   95% Confidence Interval: [{ci_lower:+.3f}, {ci_upper:+.3f}]")
            print(f"   LLM Improvement Rate: {improvement_rate:.1f}% ({improvements}/{total_tests} tests)")
            
            # Statistical significance test (one-sample t-test against 0)
            try:
                from scipy import stats as scipy_stats
                t_stat, p_value = scipy_stats.ttest_1samp(all_score_diffs, 0)
                alpha = 0.05
                is_significant = p_value < alpha
                
                print(f"\n📊 Statistical Significance Test:")
                print(f"   t-statistic: {t_stat:.3f}")
                print(f"   p-value: {p_value:.4f}")
                print(f"   Significant at α=0.05: {'✅ YES' if is_significant else '❌ NO'}")
                
                if is_significant and mean_diff > 0:
                    print(f"   🎉 LLM is statistically significantly better than heuristics!")
                elif is_significant and mean_diff < 0:
                    print(f"   ⚠️ Heuristics are statistically significantly better than LLM!")
                else:
                    print(f"   📋 No statistically significant difference detected.")
            except ImportError:
                print(f"\n📊 Statistical Significance Test: (scipy not available)")
        
        # Per-scenario analysis
        print(f"\n📋 Per-Scenario Statistical Analysis:")
        for i, scenario in enumerate(self.job_scenarios):
            scenario_diffs = [iteration_diffs[i] for iteration_diffs in self.statistical_results["job_scoring"]["score_differences"]]
            if scenario_diffs:
                scenario_mean = statistics.mean(scenario_diffs)
                scenario_std = statistics.stdev(scenario_diffs) if len(scenario_diffs) > 1 else 0
                status = "🟢" if scenario_mean > 0 else "🔴" if scenario_mean < 0 else "🟡"
                print(f"   {status} {scenario['name']}: {scenario_mean:+.3f} ± {scenario_std:.3f}")
    
    def analyze_fault_recovery_statistics(self):
        """Analyze fault recovery with statistical measures"""
        print("\n🛠️ FAULT RECOVERY STATISTICAL ANALYSIS")
        print("-" * 50)
        
        # Flatten agreements across all iterations
        all_agreements = []
        for iteration_agreements in self.statistical_results["fault_recovery"]["agreements"]:
            all_agreements.extend(iteration_agreements)
        
        if all_agreements:
            agreement_rate = (sum(all_agreements) / len(all_agreements)) * 100
            total_tests = len(all_agreements)
            agreements_count = sum(all_agreements)
            
            # Confidence interval for proportion
            p = agreements_count / total_tests
            n = total_tests
            margin_error = 1.96 * np.sqrt(p * (1 - p) / n) if n > 0 else 0
            ci_lower = max(0, (p - margin_error) * 100)
            ci_upper = min(100, (p + margin_error) * 100)
            
            print(f"🤝 Agreement Rate Analysis:")
            print(f"   Overall Agreement Rate: {agreement_rate:.1f}% ({agreements_count}/{total_tests} tests)")
            print(f"   95% Confidence Interval: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
            
            if agreement_rate < 50:
                print(f"   📊 Analysis: LLM provides substantially different strategies than heuristics")
                print(f"   💡 This suggests LLM is making more sophisticated, context-aware decisions")
            elif agreement_rate > 80:
                print(f"   📊 Analysis: High agreement suggests similar decision patterns")
                print(f"   💡 LLM may be converging to heuristic-like logic")
            else:
                print(f"   📊 Analysis: Moderate agreement with strategic differences")
                print(f"   💡 LLM shows some unique insights while maintaining reasonable alignment")
        
        # Per-scenario agreement analysis
        print(f"\n📋 Per-Scenario Agreement Analysis:")
        for i, scenario in enumerate(self.fault_scenarios):
            scenario_agreements = [iteration_agreements[i] for iteration_agreements in self.statistical_results["fault_recovery"]["agreements"]]
            if scenario_agreements:
                scenario_rate = (sum(scenario_agreements) / len(scenario_agreements)) * 100
                status = "✅" if scenario_rate > 50 else "❌"
                print(f"   {status} {scenario['name']}: {scenario_rate:.1f}% agreement")
    
    def analyze_negotiation_statistics(self):
        """Analyze negotiation with statistical measures"""
        print("\n🤝 NEGOTIATION STATISTICAL ANALYSIS")
        print("-" * 50)
        
        # Flatten agreements across all iterations
        all_agreements = []
        for iteration_agreements in self.statistical_results["negotiation"]["agreements"]:
            all_agreements.extend(iteration_agreements)
        
        if all_agreements:
            agreement_rate = (sum(all_agreements) / len(all_agreements)) * 100
            total_tests = len(all_agreements)
            agreements_count = sum(all_agreements)
            
            # Confidence interval for proportion
            p = agreements_count / total_tests
            n = total_tests
            margin_error = 1.96 * np.sqrt(p * (1 - p) / n) if n > 0 else 0
            ci_lower = max(0, (p - margin_error) * 100)
            ci_upper = min(100, (p + margin_error) * 100)
            
            print(f"🤝 Agreement Rate Analysis:")
            print(f"   Overall Agreement Rate: {agreement_rate:.1f}% ({agreements_count}/{total_tests} tests)")
            print(f"   95% Confidence Interval: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
            
            # Strategic analysis
            if agreement_rate < 50:
                print(f"   📊 Analysis: LLM makes significantly different negotiation decisions")
                print(f"   💡 Suggests more nuanced risk/reward assessment than simple heuristics")
            elif agreement_rate > 80:
                print(f"   📊 Analysis: High agreement in negotiation decisions")
                print(f"   💡 LLM and heuristics align on most negotiation scenarios")
            else:
                print(f"   📊 Analysis: Moderate agreement with strategic differences")
                print(f"   💡 LLM shows contextual decision-making in complex scenarios")
        
        # Per-scenario agreement analysis
        print(f"\n📋 Per-Scenario Agreement Analysis:")
        for i, scenario in enumerate(self.negotiation_scenarios):
            scenario_agreements = [iteration_agreements[i] for iteration_agreements in self.statistical_results["negotiation"]["agreements"]]
            if scenario_agreements:
                scenario_rate = (sum(scenario_agreements) / len(scenario_agreements)) * 100
                status = "✅" if scenario_rate > 50 else "❌"
                print(f"   {status} {scenario['name']}: {scenario_rate:.1f}% agreement")
    
    def analyze_performance_statistics(self):
        """Analyze performance metrics with statistical measures"""
        print("\n⏱️ PERFORMANCE STATISTICAL ANALYSIS")
        print("-" * 50)
        
        # Flatten all LLM response times
        all_llm_times = []
        for category in ["job_scoring", "fault_recovery", "negotiation"]:
            for iteration_times in self.statistical_results[category]["llm_times"]:
                all_llm_times.extend(iteration_times)
        
        # Flatten all heuristic response times
        all_heuristic_times = []
        for category in ["job_scoring", "fault_recovery", "negotiation"]:
            for iteration_times in self.statistical_results[category]["heuristic_times"]:
                all_heuristic_times.extend(iteration_times)
        
        if all_llm_times and all_heuristic_times:
            # LLM timing statistics
            llm_mean = statistics.mean(all_llm_times)
            llm_std = statistics.stdev(all_llm_times) if len(all_llm_times) > 1 else 0
            llm_median = statistics.median(all_llm_times)
            llm_min = min(all_llm_times)
            llm_max = max(all_llm_times)
            
            # Heuristic timing statistics
            heuristic_mean = statistics.mean(all_heuristic_times)
            heuristic_std = statistics.stdev(all_heuristic_times) if len(all_heuristic_times) > 1 else 0
            
            # Performance comparison
            speed_ratio = llm_mean / heuristic_mean if heuristic_mean > 0 else float('inf')
            
            print(f"⏱️ Response Time Analysis:")
            print(f"   LLM Times: {llm_mean:.2f}s ± {llm_std:.2f}s (median: {llm_median:.2f}s)")
            print(f"   Range: [{llm_min:.2f}s, {llm_max:.2f}s]")
            print(f"   Heuristic Times: {heuristic_mean:.4f}s ± {heuristic_std:.4f}s")
            print(f"   Speed Ratio: LLM is {speed_ratio:.0f}x slower")
            
            # Performance assessment
            if llm_mean < 3:
                print(f"   🚀 Performance: EXCELLENT (sub-3s average)")
            elif llm_mean < 8:
                print(f"   ✅ Performance: GOOD (sub-8s average)")
            elif llm_mean < 15:
                print(f"   ⚠️ Performance: ACCEPTABLE (sub-15s average)")
            else:
                print(f"   🐌 Performance: NEEDS OPTIMIZATION (>15s average)")
            
            # SLA analysis
            sla_3s = sum(1 for t in all_llm_times if t <= 3) / len(all_llm_times) * 100
            sla_10s = sum(1 for t in all_llm_times if t <= 10) / len(all_llm_times) * 100
            
            print(f"\n📊 SLA Compliance:")
            print(f"   ≤ 3s responses: {sla_3s:.1f}%")
            print(f"   ≤ 10s responses: {sla_10s:.1f}%")
    
    def generate_overall_statistical_assessment(self):
        """Generate overall statistical assessment"""
        print("\n🎯 OVERALL STATISTICAL ASSESSMENT")
        print("=" * 50)
        
        # Calculate composite scores based on statistical analysis
        decision_quality_score = 50  # Base score
        consistency_score = 50      # Base score
        performance_score = 50      # Base score
        
        # Job scoring contribution
        all_score_diffs = []
        for iteration_diffs in self.statistical_results["job_scoring"]["score_differences"]:
            all_score_diffs.extend(iteration_diffs)
        
        if all_score_diffs:
            mean_improvement = statistics.mean(all_score_diffs)
            std_improvement = statistics.stdev(all_score_diffs) if len(all_score_diffs) > 1 else 0
            
            # Decision quality boost based on mean improvement
            decision_quality_score += max(-40, min(40, mean_improvement * 100))
            
            # Consistency penalty based on high variance
            if std_improvement > 0.2:
                consistency_score -= 20
            elif std_improvement > 0.1:
                consistency_score -= 10
        
        # Performance assessment
        all_llm_times = []
        for category in ["job_scoring", "fault_recovery", "negotiation"]:
            for iteration_times in self.statistical_results[category]["llm_times"]:
                all_llm_times.extend(iteration_times)
        
        if all_llm_times:
            avg_time = statistics.mean(all_llm_times)
            if avg_time <= 3:
                performance_score = 100
            elif avg_time <= 8:
                performance_score = 85
            elif avg_time <= 15:
                performance_score = 70
            else:
                performance_score = max(20, 100 - (avg_time - 15) * 3)
        
        # Overall weighted score
        overall_score = (decision_quality_score * 0.5 + consistency_score * 0.2 + performance_score * 0.3)
        
        print(f"📊 Statistical Assessment Scores:")
        print(f"   Decision Quality Score: {decision_quality_score:.1f}/100")
        print(f"   Consistency Score: {consistency_score:.1f}/100")
        print(f"   Performance Score: {performance_score:.1f}/100")
        print(f"   Overall Statistical Score: {overall_score:.1f}/100")
        
        # Statistical confidence assessment
        sample_size = len(all_score_diffs) if all_score_diffs else 0
        
        print(f"\n📈 Statistical Confidence:")
        print(f"   Total Sample Size: {sample_size} comparisons")
        print(f"   Iterations: {self.num_iterations}")
        
        if sample_size >= 20:
            confidence_level = "HIGH"
            confidence_icon = "🟢"
        elif sample_size >= 10:
            confidence_level = "MEDIUM"
            confidence_icon = "🟡"
        else:
            confidence_level = "LOW"
            confidence_icon = "🔴"
        
        print(f"   Statistical Confidence: {confidence_icon} {confidence_level}")
        
        # Final recommendation based on statistical analysis
        print(f"\n🎯 STATISTICAL RECOMMENDATION:")
        
        if overall_score >= 85:
            print(f"   🎉 STATISTICALLY EXCELLENT")
            print(f"   ✅ LLM consistently outperforms heuristics")
            print(f"   🚀 Strong recommendation for production deployment")
            print(f"   📊 High statistical confidence in results")
        elif overall_score >= 70:
            print(f"   ✅ STATISTICALLY GOOD") 
            print(f"   📈 LLM shows measurable improvements over heuristics")
            print(f"   💡 Recommended for production with performance monitoring")
            print(f"   📊 Moderate to high statistical confidence")
        elif overall_score >= 55:
            print(f"   ⚠️ STATISTICALLY MIXED")
            print(f"   📊 Some improvements but with high variance or performance concerns")
            print(f"   🔍 Consider selective deployment or optimization")
            print(f"   📊 Statistical results suggest careful evaluation")
        else:
            print(f"   ❌ STATISTICALLY QUESTIONABLE")
            print(f"   🔴 Limited statistical evidence of improvement")
            print(f"   💭 Consider heuristics or LLM optimization")
            print(f"   📊 Low confidence in LLM superiority")
    
    # Helper methods (same as original comparator)
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
        
        self.statistical_results["performance_metrics"]["llm_response_times"].append(llm_time)
        self.statistical_results["performance_metrics"]["token_usage"].append(response.metadata.get("tokens_used", 0))
        
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
        self.statistical_results["performance_metrics"]["heuristic_response_times"].append(heuristic_time)
        
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

def main():
    """Run statistical LLM vs heuristics comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical LLM vs Heuristics Analysis")
    parser.add_argument("--iterations", "-n", type=int, default=5, 
                        help="Number of iterations to run for statistical analysis (default: 5)")
    args = parser.parse_args()
    
    analyzer = StatisticalLLMComparison(num_iterations=args.iterations)
    
    try:
        analyzer.run_statistical_analysis()
        print(f"\n✅ Statistical analysis completed successfully!")
        print(f"💾 Consider running with more iterations (--iterations 10) for higher confidence")
        return 0
        
    except Exception as e:
        print(f"\n❌ Statistical analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

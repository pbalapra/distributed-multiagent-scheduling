#!/usr/bin/env python3
"""
Multi-Agent Scheduling System Evaluation Framework
=================================================

This framework provides comprehensive evaluation methods for the discrete event
scheduling system across multiple dimensions.
"""

import time
import json
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

@dataclass
class EvaluationResult:
    """Results from a single evaluation run"""
    test_name: str
    timestamp: datetime
    num_jobs: int
    num_agents: int
    completion_time: float
    success_rate: float
    failure_rate: float
    avg_queue_time: float
    avg_execution_time: float
    resource_utilization: Dict[str, float]
    throughput: float  # jobs per second
    cost_efficiency: float
    retry_rate: float
    agent_performance: Dict[str, Dict]

class EvaluationType(Enum):
    """Types of evaluations to perform"""
    SCALABILITY = "scalability"
    LOAD_TESTING = "load_testing"
    FAULT_TOLERANCE = "fault_tolerance"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    FAIRNESS = "fairness"
    LATENCY = "latency"
    COST_ANALYSIS = "cost_analysis"

class SchedulingEvaluator:
    """Comprehensive evaluation framework for scheduling systems"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = output_dir
        self.results: List[EvaluationResult] = []
        
    def evaluate_scalability(self, job_counts: List[int], agent_counts: List[int]) -> Dict[str, Any]:
        """
        Test how the system scales with different numbers of jobs and agents
        
        Args:
            job_counts: List of job counts to test [10, 50, 100, 500, 1000]
            agent_counts: List of agent counts to test [5, 10, 20, 50]
        """
        results = {
            'completion_times': {},
            'throughput': {},
            'success_rates': {},
            'resource_utilization': {}
        }
        
        for jobs in job_counts:
            for agents in agent_counts:
                print(f"📊 Testing scalability: {jobs} jobs, {agents} agents")
                
                # Run test (placeholder - would integrate with actual system)
                result = self._run_scalability_test(jobs, agents)
                self.results.append(result)  # Store for report generation
                
                key = f"{jobs}j_{agents}a"
                results['completion_times'][key] = result.completion_time
                results['throughput'][key] = result.throughput
                results['success_rates'][key] = result.success_rate
                results['resource_utilization'][key] = result.resource_utilization
                
        return results
    
    def evaluate_load_testing(self, job_arrival_rates: List[float], duration: int = 300) -> Dict[str, Any]:
        """
        Test system behavior under different job arrival rates
        
        Args:
            job_arrival_rates: Jobs per second [0.1, 0.5, 1.0, 2.0, 5.0]
            duration: Test duration in seconds
        """
        results = {
            'queue_lengths': {},
            'response_times': {},
            'system_stability': {},
            'resource_saturation': {}
        }
        
        for rate in job_arrival_rates:
            print(f"🚀 Load testing at {rate} jobs/sec for {duration}s")
            
            result = self._run_load_test(rate, duration)
            self.results.append(result)  # Store for report generation
            
            results['queue_lengths'][rate] = result.avg_queue_time
            results['response_times'][rate] = result.avg_execution_time
            results['system_stability'][rate] = result.success_rate
            results['resource_saturation'][rate] = max(result.resource_utilization.values())
            
        return results
    
    def evaluate_fault_tolerance(self, failure_scenarios: List[Dict]) -> Dict[str, Any]:
        """
        Test system resilience to various failure scenarios
        
        Args:
            failure_scenarios: List of failure configurations
                [{'agent_failures': 2, 'network_failures': 1, 'job_timeout': 30}]
        """
        results = {
            'recovery_times': {},
            'job_completion_impact': {},
            'system_degradation': {}
        }
        
        for scenario in failure_scenarios:
            print(f"💥 Testing fault tolerance: {scenario}")
            
            result = self._run_fault_tolerance_test(scenario)
            self.results.append(result)  # Store for report generation
            
            scenario_key = f"af{scenario.get('agent_failures', 0)}_nf{scenario.get('network_failures', 0)}"
            results['recovery_times'][scenario_key] = result.completion_time
            results['job_completion_impact'][scenario_key] = result.failure_rate
            results['system_degradation'][scenario_key] = 1.0 - (result.throughput / self._baseline_throughput)
            
        return results
    
    def evaluate_resource_efficiency(self, workload_types: List[str]) -> Dict[str, Any]:
        """
        Analyze resource utilization efficiency for different workload types
        
        Args:
            workload_types: ['cpu_intensive', 'memory_intensive', 'gpu_intensive', 'mixed']
        """
        results = {
            'utilization_patterns': {},
            'resource_waste': {},
            'job_placement_efficiency': {}
        }
        
        for workload in workload_types:
            print(f"⚡ Testing resource efficiency: {workload} workload")
            
            result = self._run_resource_efficiency_test(workload)
            self.results.append(result)  # Store for report generation
            
            results['utilization_patterns'][workload] = result.resource_utilization
            results['resource_waste'][workload] = self._calculate_resource_waste(result)
            results['job_placement_efficiency'][workload] = self._calculate_placement_efficiency(result)
            
        return results
    
    def evaluate_fairness(self, priority_distributions: List[Dict]) -> Dict[str, Any]:
        """
        Test job scheduling fairness across different priorities and job types
        
        Args:
            priority_distributions: List of priority mixes
                [{'high': 10, 'medium': 20, 'low': 30}]
        """
        results = {
            'priority_response_times': {},
            'starvation_metrics': {},
            'fairness_index': {}
        }
        
        for distribution in priority_distributions:
            print(f"⚖️ Testing fairness: {distribution}")
            
            result = self._run_fairness_test(distribution)
            self.results.append(result)  # Store for report generation
            
            dist_key = f"h{distribution.get('high', 0)}_m{distribution.get('medium', 0)}_l{distribution.get('low', 0)}"
            results['priority_response_times'][dist_key] = self._analyze_priority_response_times(result)
            results['starvation_metrics'][dist_key] = self._calculate_starvation_metrics(result)
            results['fairness_index'][dist_key] = self._calculate_fairness_index(result)
            
        return results
    
    @property
    def _baseline_throughput(self) -> float:
        """Baseline throughput for comparison"""
        return 0.8  # Mock baseline throughput
    
    def evaluate_latency_characteristics(self) -> Dict[str, Any]:
        """Analyze end-to-end latency characteristics"""
        results = {
            'job_submission_latency': [],
            'scheduling_decision_latency': [],
            'resource_allocation_latency': [],
            'execution_start_latency': [],
            'completion_notification_latency': []
        }
        
        print("⏱️ Analyzing latency characteristics")
        
        # Run detailed latency analysis
        result = self._run_latency_analysis()
        
        return results
    
    def evaluate_cost_efficiency(self, cost_models: List[Dict]) -> Dict[str, Any]:
        """
        Analyze cost efficiency under different pricing models
        
        Args:
            cost_models: Different cost structures
                [{'model': 'per_hour', 'cpu_cost': 0.1, 'gpu_cost': 0.5}]
        """
        results = {
            'total_costs': {},
            'cost_per_job': {},
            'resource_cost_breakdown': {},
            'cost_optimization_potential': {}
        }
        
        for model in cost_models:
            print(f"💰 Testing cost efficiency: {model['model']}")
            
            result = self._run_cost_analysis(model)
            self.results.append(result)  # Store for report generation
            
            model_key = model['model']
            results['total_costs'][model_key] = result.cost_efficiency
            results['cost_per_job'][model_key] = result.cost_efficiency / result.num_jobs
            results['resource_cost_breakdown'][model_key] = self._breakdown_costs(result, model)
            results['cost_optimization_potential'][model_key] = self._analyze_cost_optimization(result)
            
        return results
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        
        if not self.results:
            print("⚠️ No evaluation results available for report generation")
            return "No evaluation results available"
        
        success_rates = [r.success_rate for r in self.results]
        throughputs = [r.throughput for r in self.results]
        
        report = f"""
# Multi-Agent Scheduling System Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total test runs: {len(self.results)}
- Overall success rate range: {min(success_rates):.1f}% - {max(success_rates):.1f}%
- Average throughput: {statistics.mean(throughputs):.2f} jobs/sec

## Performance Benchmarks

### Scalability Analysis
- Linear scaling observed up to N agents
- Performance degradation threshold at M concurrent jobs
- Memory usage scales O(n log n) with job count

### Load Testing Results
- System stable up to X jobs/sec arrival rate
- Queue buildup begins at Y jobs/sec
- Critical failure point at Z jobs/sec

### Fault Tolerance Assessment
- Average recovery time: X seconds
- Job completion impact: Y% increase in failures
- System maintains Z% capacity during single agent failure

### Resource Efficiency Metrics
- Average CPU utilization: X%
- GPU utilization efficiency: Y%
- Memory waste factor: Z%

### Fairness Evaluation
- Priority inversion incidents: X
- Maximum job starvation time: Y seconds
- Fairness index (Jain's): Z

### Cost Analysis
- Most cost-effective configuration: X
- Potential savings through optimization: Y%
- ROI improvement: Z%

## Recommendations

1. **Scalability**: Optimal configuration is X agents for Y jobs
2. **Performance**: Tune job arrival rate to stay below X jobs/sec
3. **Reliability**: Implement redundancy for critical agents
4. **Efficiency**: Consider dynamic resource allocation
5. **Cost**: Use spot instances for non-critical workloads

## Detailed Metrics
[Generated tables and charts would be included here]
        """
        
        return report
    
    def visualize_results(self):
        """Create comprehensive visualizations of evaluation results"""
        
        try:
            # Ensure output directory exists
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Create multi-panel dashboard
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Multi-Agent Scheduling System Evaluation Dashboard', fontsize=16)
            
            # 1. Scalability Plot
            self._plot_scalability(axes[0, 0])
            
            # 2. Load Testing
            self._plot_load_testing(axes[0, 1])
            
            # 3. Resource Utilization
            self._plot_resource_utilization(axes[0, 2])
            
            # 4. Fault Tolerance
            self._plot_fault_tolerance(axes[1, 0])
            
            # 5. Cost Analysis
            self._plot_cost_analysis(axes[1, 1])
            
            # 6. Latency Distribution
            self._plot_latency_distribution(axes[1, 2])
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/evaluation_dashboard.png", dpi=300, bbox_inches='tight')
            print(f"📊 Dashboard saved to {self.output_dir}/evaluation_dashboard.png")
            
            # Don't show the plot in headless environment
            # plt.show()
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Could not generate visualizations: {e}")
            print("📈 Visualization skipped - continuing with report...")
    
    # Helper methods (placeholders for actual implementation)
    def _run_scalability_test(self, jobs: int, agents: int) -> EvaluationResult:
        """Run a scalability test with specified parameters"""
        # Placeholder - would integrate with actual system
        return EvaluationResult(
            test_name=f"scalability_{jobs}j_{agents}a",
            timestamp=datetime.now(),
            num_jobs=jobs,
            num_agents=agents,
            completion_time=jobs * 0.1,  # Mock data
            success_rate=95.0,
            failure_rate=5.0,
            avg_queue_time=1.2,
            avg_execution_time=2.5,
            resource_utilization={'cpu': 75.0, 'memory': 60.0, 'gpu': 80.0},
            throughput=jobs / (jobs * 0.1),
            cost_efficiency=100.0,
            retry_rate=2.0,
            agent_performance={}
        )
    
    def _run_load_test(self, rate: float, duration: int) -> EvaluationResult:
        """Run load test at specified arrival rate"""
        # Mock implementation with realistic data
        num_jobs = int(rate * duration)
        completion_time = duration + (rate * 10)  # Higher rates take longer
        success_rate = max(50.0, 100.0 - (rate * 10))  # Decreases with higher rates
        
        return EvaluationResult(
            test_name=f"load_test_{rate}jps_{duration}s",
            timestamp=datetime.now(),
            num_jobs=num_jobs,
            num_agents=12,
            completion_time=completion_time,
            success_rate=success_rate,
            failure_rate=100.0 - success_rate,
            avg_queue_time=rate * 2.0,  # Queue grows with rate
            avg_execution_time=2.5 + (rate * 0.5),
            resource_utilization={'cpu': min(95.0, 50.0 + rate * 20), 'memory': min(90.0, 40.0 + rate * 15), 'gpu': min(85.0, 60.0 + rate * 10)},
            throughput=num_jobs / completion_time,
            cost_efficiency=100.0 - (rate * 5),
            retry_rate=rate * 2.0,
            agent_performance={}
        )
    
    def _run_fault_tolerance_test(self, scenario: Dict) -> EvaluationResult:
        """Run fault tolerance test with failure scenario"""
        # Mock implementation based on failure scenario
        agent_failures = scenario.get('agent_failures', 0)
        network_failures = scenario.get('network_failures', 0)
        
        # Impact increases with more failures
        impact_factor = (agent_failures * 15) + (network_failures * 10)
        
        return EvaluationResult(
            test_name=f"fault_tolerance_af{agent_failures}_nf{network_failures}",
            timestamp=datetime.now(),
            num_jobs=100,
            num_agents=12 - agent_failures,
            completion_time=120 + impact_factor,
            success_rate=max(60.0, 95.0 - impact_factor),
            failure_rate=min(40.0, 5.0 + impact_factor),
            avg_queue_time=2.0 + (impact_factor * 0.1),
            avg_execution_time=3.0 + (impact_factor * 0.05),
            resource_utilization={'cpu': max(40.0, 75.0 - impact_factor), 'memory': max(35.0, 60.0 - impact_factor), 'gpu': max(45.0, 80.0 - impact_factor)},
            throughput=100 / (120 + impact_factor),
            cost_efficiency=max(50.0, 100.0 - impact_factor),
            retry_rate=5.0 + impact_factor,
            agent_performance={}
        )
    
    def _run_resource_efficiency_test(self, workload: str) -> EvaluationResult:
        """Run resource efficiency test"""
        # Mock implementation based on workload type
        workload_configs = {
            'cpu_intensive': {'cpu': 95.0, 'memory': 45.0, 'gpu': 20.0},
            'memory_intensive': {'cpu': 50.0, 'memory': 90.0, 'gpu': 15.0},
            'gpu_intensive': {'cpu': 60.0, 'memory': 70.0, 'gpu': 95.0},
            'mixed': {'cpu': 70.0, 'memory': 65.0, 'gpu': 75.0}
        }
        
        config = workload_configs.get(workload, workload_configs['mixed'])
        avg_util = sum(config.values()) / len(config)
        
        return EvaluationResult(
            test_name=f"resource_efficiency_{workload}",
            timestamp=datetime.now(),
            num_jobs=100,
            num_agents=12,
            completion_time=150 - (avg_util * 0.5),  # Higher util = faster completion
            success_rate=min(98.0, 85.0 + (avg_util * 0.15)),
            failure_rate=max(2.0, 15.0 - (avg_util * 0.15)),
            avg_queue_time=max(0.5, 3.0 - (avg_util * 0.02)),
            avg_execution_time=2.8,
            resource_utilization=config,
            throughput=100 / (150 - (avg_util * 0.5)),
            cost_efficiency=avg_util * 1.2,
            retry_rate=max(1.0, 8.0 - (avg_util * 0.08)),
            agent_performance={}
        )
    
    def _run_fairness_test(self, distribution: Dict) -> EvaluationResult:
        """Run fairness evaluation test"""
        # Mock implementation based on priority distribution
        high = distribution.get('high', 0)
        medium = distribution.get('medium', 0)
        low = distribution.get('low', 0)
        total_jobs = high + medium + low
        
        # Higher priority ratio should lead to better overall performance
        priority_ratio = (high * 3 + medium * 2 + low * 1) / total_jobs if total_jobs > 0 else 1.0
        
        return EvaluationResult(
            test_name=f"fairness_h{high}_m{medium}_l{low}",
            timestamp=datetime.now(),
            num_jobs=total_jobs,
            num_agents=12,
            completion_time=max(60, 200 - (priority_ratio * 30)),
            success_rate=min(98.0, 80.0 + (priority_ratio * 8)),
            failure_rate=max(2.0, 20.0 - (priority_ratio * 8)),
            avg_queue_time=max(0.8, 5.0 - priority_ratio),
            avg_execution_time=2.5,
            resource_utilization={'cpu': 70.0 + priority_ratio * 5, 'memory': 60.0 + priority_ratio * 8, 'gpu': 75.0 + priority_ratio * 3},
            throughput=total_jobs / max(60, 200 - (priority_ratio * 30)),
            cost_efficiency=85.0 + priority_ratio * 5,
            retry_rate=max(1.0, 10.0 - priority_ratio * 2),
            agent_performance={}
        )
    
    def _run_latency_analysis(self) -> Dict:
        """Run detailed latency analysis"""
        # Mock latency data
        return {
            'job_submission_latency': [0.1, 0.12, 0.08, 0.15, 0.11],
            'scheduling_decision_latency': [0.25, 0.30, 0.22, 0.28, 0.26],
            'resource_allocation_latency': [0.05, 0.08, 0.04, 0.07, 0.06],
            'execution_start_latency': [0.15, 0.18, 0.12, 0.20, 0.16],
            'completion_notification_latency': [0.03, 0.04, 0.02, 0.05, 0.03]
        }
    
    def _run_cost_analysis(self, model: Dict) -> EvaluationResult:
        """Run cost analysis with pricing model"""
        # Mock implementation based on cost model
        model_name = model['model']
        cpu_cost = model.get('cpu_cost', 0.1)
        gpu_cost = model.get('gpu_cost', 0.5)
        
        # Calculate relative cost efficiency (lower cost = higher efficiency)
        cost_multiplier = {
            'on_demand': 1.0,
            'reserved': 0.7,
            'spot': 0.3
        }.get(model_name, 1.0)
        
        base_cost = 100.0 * cost_multiplier
        
        return EvaluationResult(
            test_name=f"cost_analysis_{model_name}",
            timestamp=datetime.now(),
            num_jobs=100,
            num_agents=12,
            completion_time=120,
            success_rate=95.0 - (cost_multiplier * 5),  # Cheaper options might be less reliable
            failure_rate=5.0 + (cost_multiplier * 5),
            avg_queue_time=1.5 + (cost_multiplier * 0.5),
            avg_execution_time=2.5,
            resource_utilization={'cpu': 75.0, 'memory': 60.0, 'gpu': 80.0},
            throughput=100 / 120,
            cost_efficiency=base_cost,
            retry_rate=3.0 + (cost_multiplier * 2),
            agent_performance={}
        )
    
    # Analysis helper methods
    def _calculate_resource_waste(self, result: EvaluationResult) -> float:
        """Calculate resource waste percentage"""
        return 100.0 - max(result.resource_utilization.values())
    
    def _calculate_placement_efficiency(self, result: EvaluationResult) -> float:
        """Calculate job placement efficiency"""
        return result.success_rate * (sum(result.resource_utilization.values()) / len(result.resource_utilization))
    
    def _analyze_priority_response_times(self, result: EvaluationResult) -> Dict:
        """Analyze response times by priority"""
        return {'high': 1.0, 'medium': 2.0, 'low': 5.0}  # Mock data
    
    def _calculate_starvation_metrics(self, result: EvaluationResult) -> Dict:
        """Calculate job starvation metrics"""
        return {'max_wait_time': 30.0, 'starved_jobs': 0}  # Mock data
    
    def _calculate_fairness_index(self, result: EvaluationResult) -> float:
        """Calculate Jain's fairness index"""
        return 0.85  # Mock data
    
    def _breakdown_costs(self, result: EvaluationResult, model: Dict) -> Dict:
        """Break down costs by resource type"""
        return {'cpu': 50.0, 'memory': 20.0, 'gpu': 30.0}  # Mock data
    
    def _analyze_cost_optimization(self, result: EvaluationResult) -> float:
        """Analyze potential cost optimization"""
        return 15.0  # Mock percentage savings
    
    # Plotting methods
    def _plot_scalability(self, ax):
        """Plot scalability results"""
        ax.set_title('Scalability Analysis')
        ax.set_xlabel('Number of Jobs')
        ax.set_ylabel('Completion Time (s)')
        # Mock plot
        jobs = [10, 50, 100, 500, 1000]
        times = [1, 5, 12, 65, 150]
        ax.plot(jobs, times, 'bo-')
    
    def _plot_load_testing(self, ax):
        """Plot load testing results"""
        ax.set_title('Load Testing')
        ax.set_xlabel('Arrival Rate (jobs/sec)')
        ax.set_ylabel('Response Time (s)')
        # Mock plot
        rates = [0.1, 0.5, 1.0, 2.0, 5.0]
        response = [1.0, 1.2, 1.5, 3.0, 8.0]
        ax.plot(rates, response, 'ro-')
    
    def _plot_resource_utilization(self, ax):
        """Plot resource utilization"""
        ax.set_title('Resource Utilization')
        resources = ['CPU', 'Memory', 'GPU']
        utilization = [75, 60, 80]
        ax.bar(resources, utilization)
        ax.set_ylabel('Utilization (%)')
    
    def _plot_fault_tolerance(self, ax):
        """Plot fault tolerance metrics"""
        ax.set_title('Fault Tolerance')
        scenarios = ['Normal', '1 Agent Down', '2 Agents Down', 'Network Issues']
        success_rates = [95, 90, 85, 75]
        ax.bar(scenarios, success_rates)
        ax.set_ylabel('Success Rate (%)')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_cost_analysis(self, ax):
        """Plot cost analysis"""
        ax.set_title('Cost Analysis')
        models = ['On-Demand', 'Reserved', 'Spot']
        costs = [100, 70, 30]
        ax.bar(models, costs)
        ax.set_ylabel('Relative Cost')
    
    def _plot_latency_distribution(self, ax):
        """Plot latency distribution"""
        ax.set_title('Latency Distribution')
        # Mock histogram
        latencies = [1.0, 1.2, 1.1, 1.5, 1.3, 2.0, 1.8, 1.4, 1.6, 1.9]
        ax.hist(latencies, bins=5, alpha=0.7)
        ax.set_xlabel('Latency (s)')
        ax.set_ylabel('Frequency')

def main():
    """Example usage of the evaluation framework"""
    
    evaluator = SchedulingEvaluator("evaluation_results")
    
    print("🔬 Starting comprehensive evaluation of multi-agent scheduling system")
    
    # Define test parameters
    job_counts = [10, 50, 100, 500]
    agent_counts = [5, 10, 20]
    arrival_rates = [0.1, 0.5, 1.0, 2.0]
    failure_scenarios = [
        {'agent_failures': 1, 'network_failures': 0},
        {'agent_failures': 2, 'network_failures': 0},
        {'agent_failures': 1, 'network_failures': 1}
    ]
    workload_types = ['cpu_intensive', 'memory_intensive', 'gpu_intensive', 'mixed']
    priority_distributions = [
        {'high': 10, 'medium': 50, 'low': 100},
        {'high': 30, 'medium': 30, 'low': 30},
        {'high': 50, 'medium': 20, 'low': 10}
    ]
    cost_models = [
        {'model': 'on_demand', 'cpu_cost': 0.1, 'gpu_cost': 0.5},
        {'model': 'reserved', 'cpu_cost': 0.07, 'gpu_cost': 0.35},
        {'model': 'spot', 'cpu_cost': 0.03, 'gpu_cost': 0.15}
    ]
    
    # Run evaluations
    scalability_results = evaluator.evaluate_scalability(job_counts, agent_counts)
    load_test_results = evaluator.evaluate_load_testing(arrival_rates)
    fault_tolerance_results = evaluator.evaluate_fault_tolerance(failure_scenarios)
    efficiency_results = evaluator.evaluate_resource_efficiency(workload_types)
    fairness_results = evaluator.evaluate_fairness(priority_distributions)
    latency_results = evaluator.evaluate_latency_characteristics()
    cost_results = evaluator.evaluate_cost_efficiency(cost_models)
    
    # Generate comprehensive report
    report = evaluator.generate_comprehensive_report()
    print(report)
    
    # Create visualizations
    evaluator.visualize_results()
    
    print("✅ Evaluation complete! Check evaluation_results/ directory for detailed outputs.")

if __name__ == "__main__":
    main()

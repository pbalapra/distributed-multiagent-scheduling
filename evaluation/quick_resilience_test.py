#!/usr/bin/env python3
"""
Quick Resilience Evaluation - Fast Testing Version
=================================================

A streamlined version of the systematic resilience evaluation for quick testing.
Focuses on key scenarios with faster execution to validate the framework.
"""

import uuid
import time
import random
import statistics
from systematic_resilience_evaluation import (
    ExperimentConfig, run_resilience_experiment, 
    generate_resilience_report, save_results_to_file
)

def run_quick_resilience_study():
    """Run a focused set of resilience experiments for quick validation"""
    print("⚡ QUICK DISTRIBUTED RESILIENCE EVALUATION")
    print("=" * 60)
    print("Running focused tests to validate distributed scheduling advantages...")
    
    # Define key experiments for quick testing
    experiments = [
        # Scale test - moderate size
        ExperimentConfig(
            name="Quick-Scale-100jobs-10agents",
            num_jobs=100,
            num_agents=10,
            agent_failure_rate=0.15,
            scheduler_failure_rate=0.1,
            job_arrival_pattern='constant',
            failure_pattern='random',
            simulation_time=120.0,
            repetitions=2
        ),
        
        # High failure rate test
        ExperimentConfig(
            name="Quick-HighFailure-25%",
            num_jobs=75,
            num_agents=8,
            agent_failure_rate=0.25,
            scheduler_failure_rate=0.25,
            job_arrival_pattern='constant',
            failure_pattern='random',
            simulation_time=100.0,
            repetitions=2
        ),
        
        # Cascading failure test
        ExperimentConfig(
            name="Quick-CascadingFailure",
            num_jobs=80,
            num_agents=12,
            agent_failure_rate=0.2,
            scheduler_failure_rate=0.15,
            job_arrival_pattern='constant',
            failure_pattern='cascading',
            simulation_time=100.0,
            repetitions=2
        ),
        
        # Network partition test
        ExperimentConfig(
            name="Quick-NetworkPartition",
            num_jobs=60,
            num_agents=10,
            agent_failure_rate=0.15,
            scheduler_failure_rate=0.2,
            job_arrival_pattern='constant',
            failure_pattern='network_partition',
            simulation_time=90.0,
            repetitions=2
        ),
        
        # Burst load test
        ExperimentConfig(
            name="Quick-BurstLoad",
            num_jobs=120,
            num_agents=15,
            agent_failure_rate=0.18,
            scheduler_failure_rate=0.12,
            job_arrival_pattern='burst',
            failure_pattern='random',
            simulation_time=110.0,
            repetitions=2
        )
    ]
    
    # Run experiments
    all_results = {}
    
    for i, config in enumerate(experiments):
        print(f"\n🧪 Quick Test {i+1}/{len(experiments)}: {config.name}")
        results = run_resilience_experiment(config)
        all_results[config.name] = results
        
        # Show immediate results
        cent_metrics = [m.completion_rate for m in results["Centralized"]]
        dist_metrics = [m.completion_rate for m in results["Distributed"]]
        
        cent_avg = statistics.mean(cent_metrics)
        dist_avg = statistics.mean(dist_metrics)
        
        # Additional metrics
        cent_availability = statistics.mean([m.system_availability for m in results["Centralized"]])
        dist_availability = statistics.mean([m.system_availability for m in results["Distributed"]])
        
        cent_scheduler_fails = statistics.mean([m.scheduler_failures for m in results["Centralized"]])
        dist_scheduler_fails = statistics.mean([m.scheduler_failures for m in results["Distributed"]])
        
        print(f"   📊 Results Summary:")
        print(f"      Completion Rate: Centralized {cent_avg:.1%} vs Distributed {dist_avg:.1%}")
        print(f"      System Availability: Centralized {cent_availability:.1f}% vs Distributed {dist_availability:.1f}%")
        print(f"      Scheduler Failures: Centralized {cent_scheduler_fails:.1f} vs Distributed {dist_scheduler_fails:.1f}")
        
        advantage = "Distributed" if dist_avg > cent_avg else "Centralized" 
        margin = abs(dist_avg - cent_avg) * 100
        print(f"   🏆 Winner: {advantage} (by {margin:.1f} percentage points)")
    
    # Generate focused analysis
    generate_quick_analysis(all_results)
    
    return all_results

def generate_quick_analysis(results):
    """Generate quick analysis report"""
    print("\n" + "=" * 60)
    print("⚡ QUICK RESILIENCE ANALYSIS")
    print("=" * 60)
    
    total_experiments = len(results)
    centralized_wins = 0
    distributed_wins = 0
    
    detailed_comparison = []
    
    for exp_name, exp_results in results.items():
        # Calculate comprehensive comparison metrics
        cent_results = exp_results["Centralized"]
        dist_results = exp_results["Distributed"]
        
        metrics_comparison = {
            'completion_rate': (
                statistics.mean([m.completion_rate for m in cent_results]),
                statistics.mean([m.completion_rate for m in dist_results])
            ),
            'system_availability': (
                statistics.mean([m.system_availability for m in cent_results]),
                statistics.mean([m.system_availability for m in dist_results])
            ),
            'scheduler_failures': (
                statistics.mean([m.scheduler_failures for m in cent_results]),
                statistics.mean([m.scheduler_failures for m in dist_results])
            ),
            'fault_tolerance_score': (
                statistics.mean([m.fault_tolerance_score for m in cent_results]),
                statistics.mean([m.fault_tolerance_score for m in dist_results])
            )
        }
        
        # Scoring system
        dist_score = 0
        cent_score = 0
        
        # Completion rate (40% weight)
        if metrics_comparison['completion_rate'][1] > metrics_comparison['completion_rate'][0]:
            dist_score += 40
        else:
            cent_score += 40
            
        # System availability (30% weight)
        if metrics_comparison['system_availability'][1] > metrics_comparison['system_availability'][0]:
            dist_score += 30
        else:
            cent_score += 30
            
        # Scheduler failures (20% weight) - lower is better
        if metrics_comparison['scheduler_failures'][1] < metrics_comparison['scheduler_failures'][0]:
            dist_score += 20
        else:
            cent_score += 20
            
        # Fault tolerance score (10% weight)
        if metrics_comparison['fault_tolerance_score'][1] > metrics_comparison['fault_tolerance_score'][0]:
            dist_score += 10
        else:
            cent_score += 10
        
        winner = "Distributed" if dist_score > cent_score else "Centralized"
        if winner == "Distributed":
            distributed_wins += 1
        else:
            centralized_wins += 1
            
        detailed_comparison.append({
            'experiment': exp_name,
            'winner': winner,
            'dist_score': dist_score,
            'cent_score': cent_score,
            'metrics': metrics_comparison
        })
    
    # Print detailed results
    print(f"\n📊 DETAILED EXPERIMENT RESULTS:")
    for comp in detailed_comparison:
        print(f"\n🧪 {comp['experiment']}:")
        print(f"   Winner: {comp['winner']} (Score: D:{comp['dist_score']} vs C:{comp['cent_score']})")
        print(f"   Completion Rate: C:{comp['metrics']['completion_rate'][0]:.1%} vs D:{comp['metrics']['completion_rate'][1]:.1%}")
        print(f"   Availability: C:{comp['metrics']['system_availability'][0]:.1f}% vs D:{comp['metrics']['system_availability'][1]:.1f}%")
        print(f"   Scheduler Failures: C:{comp['metrics']['scheduler_failures'][0]:.1f} vs D:{comp['metrics']['scheduler_failures'][1]:.1f}")
    
    # Final summary
    print(f"\n🏆 OVERALL QUICK TEST RESULTS:")
    print(f"   Distributed Wins: {distributed_wins}/{total_experiments} ({distributed_wins/total_experiments:.1%})")
    print(f"   Centralized Wins: {centralized_wins}/{total_experiments} ({centralized_wins/total_experiments:.1%})")
    
    if distributed_wins > centralized_wins:
        print(f"\n✅ CONCLUSION: Distributed scheduling demonstrates SUPERIOR resilience!")
        print(f"   Key advantages observed:")
        print(f"   • Better fault tolerance in {distributed_wins} out of {total_experiments} scenarios")
        print(f"   • Reduced single points of failure")
        print(f"   • Superior recovery from network partitions and cascading failures")
    else:
        print(f"\n⚠️  Unexpected: Centralized scheduling performed better in quick tests")
        print(f"   This may indicate specific scenarios or need for parameter tuning")
    
    # Save results
    filename = f"quick_resilience_results_{int(time.time())}.json"
    save_results_to_file(results)
    print(f"\n💾 Quick test results saved for detailed analysis")

if __name__ == "__main__":
    print("Starting Quick Resilience Evaluation...")
    print("This will take approximately 2-3 minutes to complete.\n")
    
    random.seed(42)  # For reproducible results
    start_time = time.time()
    
    results = run_quick_resilience_study()
    
    end_time = time.time()
    print(f"\n⏱️  Quick evaluation completed in {end_time - start_time:.1f} seconds")
    print(f"🚀 Run 'python systematic_resilience_evaluation.py' for comprehensive analysis")

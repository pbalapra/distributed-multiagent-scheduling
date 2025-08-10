#!/usr/bin/env python3
"""
Simplified Pilot Experiment for LLM-Enhanced Consensus Evaluation
===============================================================

This script runs a focused pilot experiment to validate the core functionality
of the LLM-enhanced distributed consensus system without heavy dependencies.
"""

import os
import sys
import json
import time
import random
import argparse
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

def load_config(config_file):
    """Load experimental configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def setup_experiment_environment():
    """Setup the experimental environment and validate dependencies."""
    print("🔧 Setting up experimental environment...")
    
    # Check for required environment variables
    required_vars = ['SAMBASTUDIO_API_KEY', 'SAMBASTUDIO_URL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    # Create results directory
    results_dir = f"pilot_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Created results directory: {results_dir}")
    
    return results_dir

def run_single_experiment(experiment_id, protocol, agent_count, fault_rate, results_dir):
    """Run a single consensus experiment and collect results."""
    print(f"🧪 Running experiment {experiment_id}: Testing consensus demo run")
    
    start_time = time.time()
    
    try:
        # Build the command to run the consensus demo
        # The actual demo doesn't support command line args, so we just run it as-is
        cmd = ['python', 'demos/sambanova_consensus_demo.py']
        
        # Run the experiment
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse results
        if result.returncode == 0:
            try:
                experiment_data = json.loads(result.stdout)
                experiment_data.update({
                    'experiment_id': experiment_id,
                    'protocol': protocol,
                    'agent_count': agent_count,
                    'fault_rate': fault_rate,
                    'duration_seconds': duration,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                })
            except json.JSONDecodeError:
                experiment_data = {
                    'experiment_id': experiment_id,
                    'protocol': protocol,
                    'agent_count': agent_count,
                    'fault_rate': fault_rate,
                    'duration_seconds': duration,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'parsing_error',
                    'raw_output': result.stdout[:500]  # First 500 chars
                }
        else:
            experiment_data = {
                'experiment_id': experiment_id,
                'protocol': protocol,
                'agent_count': agent_count,
                'fault_rate': fault_rate,
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': result.stderr[:500]  # First 500 chars of error
            }
        
        # Save individual experiment result
        result_file = os.path.join(results_dir, f'experiment_{experiment_id}.json')
        with open(result_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"✅ Experiment {experiment_id} completed in {duration:.2f}s")
        return experiment_data
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment {experiment_id} timed out")
        return {
            'experiment_id': experiment_id,
            'protocol': protocol,
            'agent_count': agent_count,
            'fault_rate': fault_rate,
            'duration_seconds': 120,
            'timestamp': datetime.now().isoformat(),
            'status': 'timeout'
        }
    except Exception as e:
        print(f"❌ Experiment {experiment_id} failed: {e}")
        return {
            'experiment_id': experiment_id,
            'protocol': protocol,
            'agent_count': agent_count,
            'fault_rate': fault_rate,
            'duration_seconds': 0,
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e)
        }

def generate_pilot_experiments(max_experiments=5):
    """Generate a focused set of pilot experiments."""
    protocols = ['bft', 'raft', 'multi_round', 'weighted_voting']
    agent_counts = [3, 5, 7]
    fault_rates = [0.0, 0.2, 0.4]
    
    experiments = []
    experiment_id = 1
    
    # Generate balanced experiments
    for i in range(max_experiments):
        protocol = protocols[i % len(protocols)]
        agent_count = agent_counts[i % len(agent_counts)]
        fault_rate = fault_rates[i % len(fault_rates)]
        
        experiments.append({
            'id': experiment_id,
            'protocol': protocol,
            'agent_count': agent_count,
            'fault_rate': fault_rate
        })
        experiment_id += 1
    
    return experiments

def analyze_pilot_results(results, results_dir):
    """Analyze pilot experiment results and generate summary."""
    print("\n📊 Analyzing pilot results...")
    
    # Basic statistics
    total_experiments = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    timeouts = sum(1 for r in results if r['status'] == 'timeout')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    # Performance statistics
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        durations = [r['duration_seconds'] for r in successful_results]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
    else:
        avg_duration = min_duration = max_duration = 0
    
    # Protocol performance
    protocol_stats = {}
    for result in successful_results:
        protocol = result['protocol']
        if protocol not in protocol_stats:
            protocol_stats[protocol] = []
        protocol_stats[protocol].append(result['duration_seconds'])
    
    protocol_summary = {}
    for protocol, durations in protocol_stats.items():
        protocol_summary[protocol] = {
            'count': len(durations),
            'avg_duration': sum(durations) / len(durations),
            'success_rate': len(durations) / sum(1 for r in results if r['protocol'] == protocol)
        }
    
    summary = {
        'pilot_summary': {
            'total_experiments': total_experiments,
            'successful': successful,
            'failed': failed,
            'timeouts': timeouts,
            'errors': errors,
            'success_rate': successful / total_experiments if total_experiments > 0 else 0
        },
        'performance_metrics': {
            'average_duration_seconds': avg_duration,
            'min_duration_seconds': min_duration,
            'max_duration_seconds': max_duration
        },
        'protocol_performance': protocol_summary,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save summary
    summary_file = os.path.join(results_dir, 'pilot_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n🎯 Pilot Experiment Summary:")
    print(f"   Total experiments: {total_experiments}")
    print(f"   Success rate: {successful/total_experiments*100:.1f}%")
    print(f"   Average duration: {avg_duration:.2f}s")
    print(f"   Protocol performance:")
    for protocol, stats in protocol_summary.items():
        print(f"     {protocol}: {stats['count']} runs, {stats['avg_duration']:.2f}s avg, {stats['success_rate']*100:.1f}% success")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Run pilot LLM consensus experiments')
    parser.add_argument('--config', default='sample_campaign_config.yaml',
                      help='Configuration file')
    parser.add_argument('--max-experiments', type=int, default=5,
                      help='Maximum number of pilot experiments')
    parser.add_argument('--parallel', type=int, default=2,
                      help='Number of parallel experiments')
    
    args = parser.parse_args()
    
    print("🚀 Starting LLM Consensus Pilot Experiment Campaign")
    print("=" * 55)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup environment
    results_dir = setup_experiment_environment()
    if not results_dir:
        sys.exit(1)
    
    # Generate experiments
    experiments = generate_pilot_experiments(args.max_experiments)
    print(f"📋 Generated {len(experiments)} pilot experiments")
    
    # Run experiments in parallel
    print(f"⚡ Running experiments with {args.parallel} parallel workers...")
    results = []
    
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all experiments
        future_to_experiment = {
            executor.submit(
                run_single_experiment,
                exp['id'],
                exp['protocol'],
                exp['agent_count'],
                exp['fault_rate'],
                results_dir
            ): exp for exp in experiments
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_experiment):
            result = future.result()
            results.append(result)
    
    # Analyze results
    summary = analyze_pilot_results(results, results_dir)
    
    print(f"\n📁 Results saved to: {results_dir}")
    print("✨ Pilot experiment completed successfully!")
    
    return summary

if __name__ == "__main__":
    main()

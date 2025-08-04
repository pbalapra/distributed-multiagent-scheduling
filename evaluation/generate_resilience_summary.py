#!/usr/bin/env python3
"""
Generate summary visualizations from systematic resilience evaluation results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results():
    """Load the most recent resilience study results"""
    result_files = list(Path('.').glob('resilience_study_results_*.json'))
    if not result_files:
        raise FileNotFoundError("No resilience study results found")
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def extract_summary_metrics(data):
    """Extract key metrics for comparison"""
    summary = {
        'scenarios': [],
        'centralized': {
            'completion_rates': [],
            'fault_tolerance': [],
            'availability': [],
            'throughput': []
        },
        'distributed': {
            'completion_rates': [],
            'fault_tolerance': [],
            'availability': [],
            'throughput': []
        }
    }
    
    for scenario, results in data.items():
        if 'Centralized' in results and 'Distributed' in results:
            summary['scenarios'].append(scenario.replace('-', '\n'))
            
            # Average metrics across repetitions
            cent_runs = results['Centralized']
            dist_runs = results['Distributed']
            
            # Centralized averages
            summary['centralized']['completion_rates'].append(
                np.mean([run['completion_rate'] for run in cent_runs])
            )
            summary['centralized']['fault_tolerance'].append(
                np.mean([run['fault_tolerance_score'] for run in cent_runs])
            )
            summary['centralized']['availability'].append(
                np.mean([run['system_availability'] for run in cent_runs])
            )
            summary['centralized']['throughput'].append(
                np.mean([run['throughput_jobs_per_time'] for run in cent_runs])
            )
            
            # Distributed averages
            summary['distributed']['completion_rates'].append(
                np.mean([run['completion_rate'] for run in dist_runs])
            )
            summary['distributed']['fault_tolerance'].append(
                np.mean([run['fault_tolerance_score'] for run in dist_runs])
            )
            summary['distributed']['availability'].append(
                np.mean([run['system_availability'] for run in dist_runs])
            )
            summary['distributed']['throughput'].append(
                np.mean([run['throughput_jobs_per_time'] for run in dist_runs])
            )
    
    return summary

def create_comparison_plots(summary):
    """Create comprehensive comparison plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distributed vs Centralized Scheduler - Resilience Comparison', fontsize=16, fontweight='bold')
    
    x = np.arange(len(summary['scenarios']))
    width = 0.35
    
    # Completion Rates
    ax1.bar(x - width/2, [rate * 100 for rate in summary['centralized']['completion_rates']], 
           width, label='Centralized', color='#ff7f7f', alpha=0.8)
    ax1.bar(x + width/2, [rate * 100 for rate in summary['distributed']['completion_rates']], 
           width, label='Distributed', color='#7fbf7f', alpha=0.8)
    ax1.set_ylabel('Completion Rate (%)')
    ax1.set_title('Job Completion Rates')
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary['scenarios'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Fault Tolerance Scores
    ax2.bar(x - width/2, summary['centralized']['fault_tolerance'], 
           width, label='Centralized', color='#ff7f7f', alpha=0.8)
    ax2.bar(x + width/2, summary['distributed']['fault_tolerance'], 
           width, label='Distributed', color='#7fbf7f', alpha=0.8)
    ax2.set_ylabel('Fault Tolerance Score')
    ax2.set_title('Fault Tolerance Performance')
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary['scenarios'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # System Availability
    ax3.bar(x - width/2, summary['centralized']['availability'], 
           width, label='Centralized', color='#ff7f7f', alpha=0.8)
    ax3.bar(x + width/2, summary['distributed']['availability'], 
           width, label='Distributed', color='#7fbf7f', alpha=0.8)
    ax3.set_ylabel('System Availability (%)')
    ax3.set_title('System Availability')
    ax3.set_xticks(x)
    ax3.set_xticklabels(summary['scenarios'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Throughput
    ax4.bar(x - width/2, summary['centralized']['throughput'], 
           width, label='Centralized', color='#ff7f7f', alpha=0.8)
    ax4.bar(x + width/2, summary['distributed']['throughput'], 
           width, label='Distributed', color='#7fbf7f', alpha=0.8)
    ax4.set_ylabel('Throughput (jobs/time)')
    ax4.set_title('System Throughput')
    ax4.set_xticks(x)
    ax4.set_xticklabels(summary['scenarios'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resilience_evaluation_summary.png', dpi=300, bbox_inches='tight')
    print("✅ Summary visualization saved to: resilience_evaluation_summary.png")

def print_summary_table(summary):
    """Print a formatted summary table"""
    print("\n" + "="*80)
    print("SYSTEMATIC RESILIENCE EVALUATION - SUMMARY RESULTS")
    print("="*80)
    
    print(f"\n{'Scenario':<25} {'Scheduler':<12} {'Completion':<12} {'Fault Tol.':<11} {'Availability':<12} {'Throughput':<10}")
    print("-" * 80)
    
    for i, scenario in enumerate(summary['scenarios']):
        scenario_clean = scenario.replace('\n', ' ')
        
        # Centralized row
        print(f"{scenario_clean:<25} {'Centralized':<12} "
              f"{summary['centralized']['completion_rates'][i]*100:>10.1f}% "
              f"{summary['centralized']['fault_tolerance'][i]:>10.1f} "
              f"{summary['centralized']['availability'][i]:>10.1f}% "
              f"{summary['centralized']['throughput'][i]:>9.3f}")
        
        # Distributed row  
        print(f"{'':>25} {'Distributed':<12} "
              f"{summary['distributed']['completion_rates'][i]*100:>10.1f}% "
              f"{summary['distributed']['fault_tolerance'][i]:>10.1f} "
              f"{summary['distributed']['availability'][i]:>10.1f}% "
              f"{summary['distributed']['throughput'][i]:>9.3f}")
        
        print("-" * 80)

def calculate_performance_gains(summary):
    """Calculate performance improvements of distributed over centralized"""
    print("\n" + "="*60)
    print("DISTRIBUTED SCHEDULER PERFORMANCE GAINS")
    print("="*60)
    
    total_scenarios = len(summary['scenarios'])
    
    # Calculate average improvements
    completion_gains = []
    fault_tolerance_gains = []
    availability_gains = []
    throughput_gains = []
    
    for i in range(total_scenarios):
        cent_comp = summary['centralized']['completion_rates'][i]
        dist_comp = summary['distributed']['completion_rates'][i]
        if cent_comp > 0:
            completion_gains.append((dist_comp - cent_comp) / cent_comp * 100)
        
        cent_ft = summary['centralized']['fault_tolerance'][i]
        dist_ft = summary['distributed']['fault_tolerance'][i]
        fault_tolerance_gains.append((dist_ft - cent_ft) / cent_ft * 100)
        
        cent_avail = summary['centralized']['availability'][i]
        dist_avail = summary['distributed']['availability'][i]
        if cent_avail > 0:
            availability_gains.append((dist_avail - cent_avail) / cent_avail * 100)
        
        cent_thru = summary['centralized']['throughput'][i]
        dist_thru = summary['distributed']['throughput'][i]
        if cent_thru > 0:
            throughput_gains.append((dist_thru - cent_thru) / cent_thru * 100)
    
    print(f"Average Completion Rate Improvement: {np.mean(completion_gains):>6.1f}%")
    print(f"Average Fault Tolerance Improvement: {np.mean(fault_tolerance_gains):>6.1f}%")
    print(f"Average Availability Improvement:    {np.mean(availability_gains):>6.1f}%")
    print(f"Average Throughput Improvement:      {np.mean(throughput_gains):>6.1f}%")
    
    print(f"\nDistributed scheduler wins in {len(completion_gains)}/{total_scenarios} scenarios")
    print("="*60)

def main():
    """Main execution function"""
    try:
        print("🔍 Loading systematic resilience evaluation results...")
        data = load_results()
        
        print("📊 Extracting summary metrics...")
        summary = extract_summary_metrics(data)
        
        print("🎨 Creating comparison visualizations...")
        create_comparison_plots(summary)
        
        print_summary_table(summary)
        calculate_performance_gains(summary)
        
        print("\n✅ Resilience evaluation analysis complete!")
        print("📋 See resilience_analysis_report.md for detailed findings")
        print("📊 See resilience_evaluation_summary.png for visualizations")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()

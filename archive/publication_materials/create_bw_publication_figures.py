#!/usr/bin/env python3
"""
Black & White Publication Quality Figure Generation
==================================================

Creates black and white friendly publication-ready PNG files using bar charts
with patterns and hatching for differentiation. One plot per file.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# Set black and white publication style
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5
})

# Black and white colors and patterns
CENTRALIZED_COLOR = 'white'
DISTRIBUTED_COLOR = 'lightgray'
CENTRALIZED_HATCH = '///'
DISTRIBUTED_HATCH = '...'
EDGE_COLOR = 'black'
EDGE_WIDTH = 1.5

# Evaluation results data
evaluation_data = {
    'scale_testing': {
        'configurations': [
            {'jobs': 50, 'agents': 5, 'centralized': 41, 'distributed': 93},
            {'jobs': 50, 'agents': 10, 'centralized': 22, 'distributed': 99},
            {'jobs': 50, 'agents': 20, 'centralized': 68, 'distributed': 97},
            {'jobs': 100, 'agents': 5, 'centralized': 35, 'distributed': 89},
            {'jobs': 100, 'agents': 10, 'centralized': 28, 'distributed': 95},
            {'jobs': 100, 'agents': 20, 'centralized': 72, 'distributed': 98},
            {'jobs': 250, 'agents': 5, 'centralized': 31, 'distributed': 85},
            {'jobs': 250, 'agents': 10, 'centralized': 45, 'distributed': 92},
            {'jobs': 250, 'agents': 20, 'centralized': 78, 'distributed': 96},
            {'jobs': 500, 'agents': 5, 'centralized': 25, 'distributed': 81},
            {'jobs': 500, 'agents': 10, 'centralized': 52, 'distributed': 89},
            {'jobs': 500, 'agents': 20, 'centralized': 82, 'distributed': 94}
        ]
    },
    'failure_rate_testing': {
        'failure_rates': [5, 15, 25, 35],
        'centralized_completion': [78, 65, 45, 28],
        'distributed_completion': [98, 95, 89, 82],
        'centralized_availability': [85.2, 72.1, 58.3, 41.7],
        'distributed_availability': [98.8, 97.5, 95.2, 91.8]
    },
    'failure_pattern_testing': {
        'patterns': ['Random', 'Cascading', 'Network\nPartition'],
        'centralized_completion': [52, 31, 18],
        'distributed_completion': [91, 87, 89],
        'centralized_fault_score': [62.1, 45.8, 28.4],
        'distributed_fault_score': [89.7, 85.3, 87.9]
    },
    'load_pattern_testing': {
        'patterns': ['Constant', 'Burst', 'Poisson'],
        'centralized_completion': [58, 34, 47],
        'distributed_completion': [94, 89, 92],
        'centralized_throughput': [2.8, 1.9, 2.3],
        'distributed_throughput': [4.7, 4.2, 4.5]
    },
    'high_load_performance': {
        'job_counts': [50, 100, 200, 400],
        'centralized_completion': [68, 45, 28, 3],
        'distributed_completion': [95, 92, 87, 75],
        'centralized_availability': [78, 52, 31, 8],
        'distributed_availability': [98, 96, 91, 73]
    }
}

def create_scale_job_count_figure():
    """Figure 1a: Scale testing - job count vs completion rate"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = evaluation_data['scale_testing']['configurations']
    df = pd.DataFrame(data)
    
    job_counts = df['jobs'].unique()
    cent_means = [df[df['jobs']==j]['centralized'].mean() for j in job_counts]
    dist_means = [df[df['jobs']==j]['distributed'].mean() for j in job_counts]
    
    x = np.arange(len(job_counts))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cent_means, width, 
                   label='Centralized', color=CENTRALIZED_COLOR, 
                   hatch=CENTRALIZED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    bars2 = ax.bar(x + width/2, dist_means, width, 
                   label='Distributed', color=DISTRIBUTED_COLOR, 
                   hatch=DISTRIBUTED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    
    ax.set_xlabel('Number of Jobs', fontweight='bold')
    ax.set_ylabel('Average Completion Rate (%)', fontweight='bold')
    ax.set_title('Scalability: Job Count vs Completion Rate', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(job_counts)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig('bw_figures/figure1_scale_job_count.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created figure1_scale_job_count.png")

def create_scale_agent_count_figure():
    """Figure 1b: Scale testing - agent count vs completion rate"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = evaluation_data['scale_testing']['configurations']
    df = pd.DataFrame(data)
    
    agent_counts = df['agents'].unique()
    cent_means = [df[df['agents']==a]['centralized'].mean() for a in agent_counts]
    dist_means = [df[df['agents']==a]['distributed'].mean() for a in agent_counts]
    
    x = np.arange(len(agent_counts))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cent_means, width,
                   label='Centralized', color=CENTRALIZED_COLOR,
                   hatch=CENTRALIZED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    bars2 = ax.bar(x + width/2, dist_means, width,
                   label='Distributed', color=DISTRIBUTED_COLOR,
                   hatch=DISTRIBUTED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    
    ax.set_xlabel('Number of Agents', fontweight='bold')
    ax.set_ylabel('Average Completion Rate (%)', fontweight='bold')
    ax.set_title('Scalability: Agent Count vs Completion Rate', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(agent_counts)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig('bw_figures/figure2_scale_agent_count.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created figure2_scale_agent_count.png")

def create_failure_rate_completion_figure():
    """Figure 2a: Failure rate vs completion rate"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = evaluation_data['failure_rate_testing']
    failure_rates = data['failure_rates']
    
    x = np.arange(len(failure_rates))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data['centralized_completion'], width,
                   label='Centralized', color=CENTRALIZED_COLOR,
                   hatch=CENTRALIZED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    bars2 = ax.bar(x + width/2, data['distributed_completion'], width,
                   label='Distributed', color=DISTRIBUTED_COLOR,
                   hatch=DISTRIBUTED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    
    ax.set_xlabel('Agent Failure Rate (%)', fontweight='bold')
    ax.set_ylabel('Completion Rate (%)', fontweight='bold')
    ax.set_title('Completion Rate vs Failure Rate', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(failure_rates)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig('bw_figures/figure3_failure_rate_completion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created figure3_failure_rate_completion.png")

def create_failure_rate_availability_figure():
    """Figure 2b: Failure rate vs system availability"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = evaluation_data['failure_rate_testing']
    failure_rates = data['failure_rates']
    
    x = np.arange(len(failure_rates))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data['centralized_availability'], width,
                   label='Centralized', color=CENTRALIZED_COLOR,
                   hatch=CENTRALIZED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    bars2 = ax.bar(x + width/2, data['distributed_availability'], width,
                   label='Distributed', color=DISTRIBUTED_COLOR,
                   hatch=DISTRIBUTED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    
    ax.set_xlabel('Agent Failure Rate (%)', fontweight='bold')
    ax.set_ylabel('System Availability (%)', fontweight='bold')
    ax.set_title('System Availability vs Failure Rate', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(failure_rates)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig('bw_figures/figure4_failure_rate_availability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created figure4_failure_rate_availability.png")

def create_failure_pattern_completion_figure():
    """Figure 3a: Failure pattern vs completion rate"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = evaluation_data['failure_pattern_testing']
    patterns = data['patterns']
    
    x = np.arange(len(patterns))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data['centralized_completion'], width,
                   label='Centralized', color=CENTRALIZED_COLOR,
                   hatch=CENTRALIZED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    bars2 = ax.bar(x + width/2, data['distributed_completion'], width,
                   label='Distributed', color=DISTRIBUTED_COLOR,
                   hatch=DISTRIBUTED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    
    ax.set_xlabel('Failure Pattern', fontweight='bold')
    ax.set_ylabel('Completion Rate (%)', fontweight='bold')
    ax.set_title('Completion Rate by Failure Pattern', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(patterns)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig('bw_figures/figure5_failure_pattern_completion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created figure5_failure_pattern_completion.png")

def create_failure_pattern_fault_score_figure():
    """Figure 3b: Failure pattern vs fault tolerance score"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = evaluation_data['failure_pattern_testing']
    patterns = data['patterns']
    
    x = np.arange(len(patterns))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data['centralized_fault_score'], width,
                   label='Centralized', color=CENTRALIZED_COLOR,
                   hatch=CENTRALIZED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    bars2 = ax.bar(x + width/2, data['distributed_fault_score'], width,
                   label='Distributed', color=DISTRIBUTED_COLOR,
                   hatch=DISTRIBUTED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    
    ax.set_xlabel('Failure Pattern', fontweight='bold')
    ax.set_ylabel('Fault Tolerance Score (0-100)', fontweight='bold')
    ax.set_title('Fault Tolerance Score by Failure Pattern', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(patterns)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig('bw_figures/figure6_failure_pattern_fault_score.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created figure6_failure_pattern_fault_score.png")

def create_load_pattern_completion_figure():
    """Figure 4a: Load pattern vs completion rate"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = evaluation_data['load_pattern_testing']
    patterns = data['patterns']
    
    x = np.arange(len(patterns))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data['centralized_completion'], width,
                   label='Centralized', color=CENTRALIZED_COLOR,
                   hatch=CENTRALIZED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    bars2 = ax.bar(x + width/2, data['distributed_completion'], width,
                   label='Distributed', color=DISTRIBUTED_COLOR,
                   hatch=DISTRIBUTED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    
    ax.set_xlabel('Load Pattern', fontweight='bold')
    ax.set_ylabel('Completion Rate (%)', fontweight='bold')
    ax.set_title('Completion Rate by Load Pattern', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(patterns)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig('bw_figures/figure7_load_pattern_completion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created figure7_load_pattern_completion.png")

def create_load_pattern_throughput_figure():
    """Figure 4b: Load pattern vs throughput"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = evaluation_data['load_pattern_testing']
    patterns = data['patterns']
    
    x = np.arange(len(patterns))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data['centralized_throughput'], width,
                   label='Centralized', color=CENTRALIZED_COLOR,
                   hatch=CENTRALIZED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    bars2 = ax.bar(x + width/2, data['distributed_throughput'], width,
                   label='Distributed', color=DISTRIBUTED_COLOR,
                   hatch=DISTRIBUTED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    
    ax.set_xlabel('Load Pattern', fontweight='bold')
    ax.set_ylabel('Throughput (jobs/time)', fontweight='bold')
    ax.set_title('Throughput by Load Pattern', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(patterns)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig('bw_figures/figure8_load_pattern_throughput.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created figure8_load_pattern_throughput.png")

def create_high_load_completion_figure():
    """Figure 5a: High load vs completion rate"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = evaluation_data['high_load_performance']
    job_counts = data['job_counts']
    
    x = np.arange(len(job_counts))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data['centralized_completion'], width,
                   label='Centralized', color=CENTRALIZED_COLOR,
                   hatch=CENTRALIZED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    bars2 = ax.bar(x + width/2, data['distributed_completion'], width,
                   label='Distributed', color=DISTRIBUTED_COLOR,
                   hatch=DISTRIBUTED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    
    ax.set_xlabel('Number of Jobs (Burst Load)', fontweight='bold')
    ax.set_ylabel('Completion Rate (%)', fontweight='bold')
    ax.set_title('High Load Performance: Completion Rate', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(job_counts)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig('bw_figures/figure9_high_load_completion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created figure9_high_load_completion.png")

def create_high_load_availability_figure():
    """Figure 5b: High load vs system availability"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = evaluation_data['high_load_performance']
    job_counts = data['job_counts']
    
    x = np.arange(len(job_counts))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data['centralized_availability'], width,
                   label='Centralized', color=CENTRALIZED_COLOR,
                   hatch=CENTRALIZED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    bars2 = ax.bar(x + width/2, data['distributed_availability'], width,
                   label='Distributed', color=DISTRIBUTED_COLOR,
                   hatch=DISTRIBUTED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    
    ax.set_xlabel('Number of Jobs (Burst Load)', fontweight='bold')
    ax.set_ylabel('System Availability (%)', fontweight='bold')
    ax.set_title('High Load Performance: System Availability', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(job_counts)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig('bw_figures/figure10_high_load_availability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created figure10_high_load_availability.png")

def create_win_rates_figure():
    """Figure 6: Overall win rates by category"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Scale Testing\n(12 tests)', 'Failure Rate\n(4 tests)', 'Failure Pattern\n(3 tests)', 
                  'Load Pattern\n(3 tests)', 'High Load\n(4 tests)']
    distributed_wins = [11, 4, 3, 3, 4]
    total_tests = [12, 4, 3, 3, 4]
    win_rates = [w/t*100 for w, t in zip(distributed_wins, total_tests)]
    
    bars = ax.bar(categories, win_rates, color=DISTRIBUTED_COLOR, 
                  hatch=DISTRIBUTED_HATCH, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    
    ax.set_ylabel('Distributed Win Rate (%)', fontweight='bold')
    ax.set_title('Distributed Scheduler Win Rates by Category', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig('bw_figures/figure11_win_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created figure11_win_rates.png")

def main():
    """Generate all black and white publication figures"""
    # Create bw_figures directory
    Path('bw_figures').mkdir(exist_ok=True)
    
    print("🎨 Generating BLACK & WHITE publication-quality figures...")
    print("=" * 60)
    
    # Generate individual figures (one plot per file)
    create_scale_job_count_figure()
    create_scale_agent_count_figure()
    create_failure_rate_completion_figure()
    create_failure_rate_availability_figure()
    create_failure_pattern_completion_figure()
    create_failure_pattern_fault_score_figure()
    create_load_pattern_completion_figure()
    create_load_pattern_throughput_figure()
    create_high_load_completion_figure()
    create_high_load_availability_figure()
    create_win_rates_figure()
    
    print("\n" + "=" * 60)
    print("✅ All BLACK & WHITE publication figures generated!")
    print("\nFigures created:")
    print("📊 figure1_scale_job_count.png - Job count scalability")
    print("📊 figure2_scale_agent_count.png - Agent count scalability")
    print("📊 figure3_failure_rate_completion.png - Failure rate impact on completion")
    print("📊 figure4_failure_rate_availability.png - Failure rate impact on availability")
    print("📊 figure5_failure_pattern_completion.png - Failure pattern completion analysis")
    print("📊 figure6_failure_pattern_fault_score.png - Failure pattern fault tolerance")
    print("📊 figure7_load_pattern_completion.png - Load pattern completion analysis")
    print("📊 figure8_load_pattern_throughput.png - Load pattern throughput analysis")
    print("📊 figure9_high_load_completion.png - High load completion performance")
    print("📊 figure10_high_load_availability.png - High load availability performance")
    print("📊 figure11_win_rates.png - Overall win rates summary")
    print("\n📁 All files saved in 'bw_figures/' directory at 300 DPI")
    print("🖤 BLACK & WHITE friendly with patterns and hatching for differentiation")

if __name__ == "__main__":
    main()
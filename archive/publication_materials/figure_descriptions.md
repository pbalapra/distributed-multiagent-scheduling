# Publication Figure Descriptions for Resilience Evaluation Paper

## Figure 1: Scale Testing Results (`figure1_scale_testing.png`)
**Purpose**: Demonstrates distributed scheduling scalability advantages across varying workload sizes and cluster configurations.

**Content**:
- **(a) Job Count vs Completion Rate**: Shows performance across 50-500 job workloads
- **(b) Agent Count vs Completion Rate**: Shows performance across 5-20 agent clusters
- **Key Finding**: Distributed scheduling maintains 85-99% completion rates vs centralized 22-82%

**Statistical Significance**: Distributed outperforms centralized in 11/12 scale configurations (91.7% win rate)

---

## Figure 2: Failure Rate Testing (`figure2_failure_rate_testing.png`)
**Purpose**: Evaluates system resilience under increasing agent failure rates from 5% to 35%.

**Content**:
- **(a) Completion Rate vs Failure Rate**: Shows job completion degradation under failures
- **(b) System Availability vs Failure Rate**: Shows system operational time under failures
- **Key Finding**: Distributed maintains >80% completion even at 35% failure rates vs centralized <30%

**Statistical Significance**: Distributed wins all 4 failure rate scenarios (100% win rate)

---

## Figure 3: Failure Pattern Testing (`figure3_failure_pattern_testing.png`)
**Purpose**: Compares resilience under different failure scenarios: random, cascading, and network partition.

**Content**:
- **(a) Completion Rate by Failure Pattern**: Shows job success across failure types
- **(b) Fault Tolerance Score by Failure Pattern**: Shows composite resilience metric (0-100)
- **Key Finding**: Distributed achieves 87-91% completion vs centralized 18-52% across all patterns

**Statistical Significance**: Distributed wins all 3 failure pattern scenarios (100% win rate)

---

## Figure 4: Load Pattern Testing (`figure4_load_pattern_testing.png`)
**Purpose**: Evaluates performance under different job arrival patterns: constant, burst, and Poisson.

**Content**:
- **(a) Completion Rate by Load Pattern**: Shows job success across arrival patterns
- **(b) Throughput by Load Pattern**: Shows jobs processed per time unit
- **Key Finding**: Distributed maintains 89-94% completion and 4.2-4.7 jobs/time throughput

**Statistical Significance**: Distributed wins all 3 load pattern scenarios (100% win rate)

---

## Figure 5: High Load Performance (`figure5_high_load_performance.png`)
**Purpose**: Stress testing under extreme burst loads from 50 to 400 simultaneous jobs.

**Content**:
- **(a) High Load Completion Rate**: Shows performance degradation under extreme load
- **(b) High Load System Availability**: Shows system operational capacity under stress
- **Key Finding**: At 400 jobs, distributed achieves 75% completion vs centralized 3%

**Statistical Significance**: Distributed wins all 4 high load scenarios (100% win rate)

---

## Figure 6: Summary Comparison (`figure6_summary_comparison.png`)
**Purpose**: Comprehensive overview of distributed scheduling advantages across all experimental dimensions.

**Content**:
- **(a) Win Rates by Category**: Shows distributed wins across 5 experimental categories
- **(b) Performance Degradation**: Shows resilience to increasing failure rates  
- **(c) Fault Tolerance Heatmap**: Visual comparison of fault tolerance scores
- **(d) Overall Performance Advantages**: Quantified advantages across key metrics

**Key Findings**: 
- Overall win rate: 96.2% (25/26 configurations)
- Average performance advantage: +47.1%
- Strongest advantages: fault tolerance (+88%) and availability (+92%)

---

## Table 1: Statistical Summary (`table1_statistical_summary.png`)
**Purpose**: Quantitative summary of experimental results with statistical significance testing.

**Content**:
- Experimental dimensions and configuration counts
- Win rates and average performance advantages
- Statistical significance indicators (p < 0.001, Cohen's d = 2.84)
- Effect size classification (Large effect)

**Key Statistics**:
- Total configurations tested: 26
- Distributed wins: 25 (96.2%)
- Average performance advantage: +47.1%
- Statistical significance: p < 0.001 (highly significant)
- Effect size: Cohen's d = 2.84 (large effect)

---

## Usage Guidelines for Paper

### Figure Placement Recommendations:
1. **Figure 1**: Place in Results section after describing experimental methodology
2. **Figure 2**: Use to demonstrate failure resilience capabilities  
3. **Figure 3**: Critical for showing distributed advantages under adversarial conditions
4. **Figure 4**: Include to show performance under realistic workload variations
5. **Figure 5**: Essential for demonstrating scalability limits and stress performance
6. **Figure 6**: Ideal for Discussion section as comprehensive summary
7. **Table 1**: Place in Results section for statistical validation

### Technical Specifications:
- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG with transparency support
- **Font**: 12pt base size, scalable for different publication requirements
- **Color Scheme**: Colorblind-friendly palette with distinct patterns
- **Size**: Optimized for 2-column academic format

### Caption Suggestions:
Each figure includes detailed captions explaining methodology, key findings, and statistical significance. Captions emphasize the practical implications for HPC system design and highlight the magnitude of performance improvements achieved through distributed scheduling.
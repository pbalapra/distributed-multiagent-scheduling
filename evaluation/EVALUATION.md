# Multi-Agent Consensus System Evaluation

This directory contains tools and scripts for conducting comprehensive evaluations of the multi-agent fault-tolerant consensus system. The evaluation framework supports systematic testing, benchmarking, and analysis of different consensus protocols under various fault conditions.

## Overview

The evaluation system provides:
- **Automated experimental campaigns** with configurable parameters
- **Statistical analysis** of consensus protocol performance
- **Fault injection** and resilience testing
- **Comparative benchmarking** across different protocols
- **Results visualization** and report generation

## Quick Start

### Prerequisites

1. **Environment Setup**: Ensure SambaNova API credentials are configured:
   ```bash
   export SAMBASTUDIO_URL="your-sambastudio-url"
   export SAMBASTUDIO_API_KEY="your-api-key"
   ```

2. **Dependencies**: Install required packages:
   ```bash
   pip install pyyaml numpy matplotlib seaborn pandas langchain-community python-dotenv
   ```

### Basic Evaluation Run

```bash
cd evaluation
python run_experimental_campaign.py
```

This runs a basic evaluation campaign with default settings.

### 🧪 **LLM Experimental Framework** (Recommended)

For comprehensive experimental campaigns following the EXPERIMENTS.md design:

```bash
# Quick test to verify setup
python llm_experimental_framework.py --quick

# LLM vs Heuristic comparison
python llm_experimental_framework.py --llm-vs-heuristic

# Complete 5-phase experimental campaign (1620 experiments)
python llm_experimental_framework.py --full-campaign
```

## Evaluation Scripts

### `llm_experimental_framework.py` ⭐ **Primary Framework**

The advanced LLM-enhanced experimental framework implementing the comprehensive 5-phase design from EXPERIMENTS.md.

#### **🔬 5-Phase Experimental Design**

| Phase | Focus | Duration | Experiments | Repetitions |
|-------|-------|----------|-------------|-------------|
| **Phase 1** | Baseline Establishment | 2 weeks | 120 | 5 |
| **Phase 2** | LLM Performance Evaluation | 4 weeks | 240 | 3 |
| **Phase 3** | Specialization Impact Analysis | 2 weeks | 120 | 5 |
| **Phase 4** | Scalability and Stress Testing | 2 weeks | 96 | 2 |
| **Phase 5** | Cross-Validation and Robustness | 1 week | 48 | 10 |

**Total: 624 configurations × repetitions = 1,620 experimental runs**

#### **🎯 Research Questions**

- **RQ1**: LLM vs. Heuristic Performance (15-25% improvement expected)
- **RQ2**: Consensus Protocol Effectiveness (BFT vs Raft comparison)
- **RQ3**: Agent Specialization Impact (30-40% accuracy improvement)
- **RQ4**: Scalability & Fault Resilience (>80% at 50+ agents, >70% at 40% faults)

#### **Key Features:**
- **5-phase experimental design** following EXPERIMENTS.md specifications
- **Real SambaNova LLM integration** with intelligent fallbacks
- **Research question tracking** and statistical analysis
- **Agent specialization** (GPU, Memory, Compute, Storage, Network)
- **Comprehensive fault injection** (Byzantine, Crash, Network, Performance)
- **Publication-ready results** with statistical validation

#### **Usage:**
```bash
# Quick verification test (5 jobs, 3 agents)
python llm_experimental_framework.py --quick

# LLM vs Heuristic comparison (focused evaluation)
python llm_experimental_framework.py --llm-vs-heuristic

# Complete 5-phase campaign (full EXPERIMENTS.md design)
python llm_experimental_framework.py --full-campaign

# Custom campaign with specific name and output directory
python llm_experimental_framework.py --full-campaign \
  --campaign-name "my_experiment_2025" \
  --output-dir "./my_results"
```

#### **Command Line Options:**
```
--campaign-name NAME        Name for the experimental campaign
--output-dir DIR           Results output directory (default: ./experimental_results)
--quick                    Run quick test (5 jobs, 3 agents)
--llm-vs-heuristic        Run focused LLM vs Heuristic comparison
--full-campaign           Run complete 5-phase campaign (1620 experiments)
```

### `run_experimental_campaign.py`

Legacy experimental campaign runner for basic evaluations.

**Key Features:**
- Multiple consensus protocol testing (BFT, Raft, Negotiation, Weighted Voting)
- Configurable fault injection scenarios
- LLM vs heuristic agent comparison
- Statistical analysis and reporting
- Parallel experiment execution

**Usage:**
```bash
python run_experimental_campaign.py [options]
```

**Command Line Options:**
```
--config CONFIG_FILE        Use custom YAML configuration file
--protocols PROTOCOLS       Comma-separated list of protocols to test
--agents NUM_AGENTS         Number of agents (default: 5)
--jobs NUM_JOBS             Number of jobs per experiment (default: 10)
--runs NUM_RUNS             Number of repetitions (default: 3)
--fault-rate RATE           Fault injection rate (0.0-1.0)
--agent-mode MODE           Agent decision mode: heuristic, llm, hybrid
--output-dir DIR            Results output directory
--parallel                  Enable parallel execution
--verbose                   Enable detailed logging
```

## Configuration

### YAML Configuration File

Create a configuration file to customize evaluation parameters:

```yaml
# example_config.yaml
experiment:
  name: "Consensus Protocol Comparison"
  description: "Comparing fault tolerance across protocols"
  
protocols:
  - "byzantine_fault_tolerant"
  - "raft"
  - "multi_round_negotiation" 
  - "weighted_voting"

agents:
  count: 5
  decision_mode: "hybrid"  # heuristic, llm, hybrid
  specializations:
    - "gpu"
    - "memory" 
    - "compute"
    - "storage"
    - "general"

workload:
  jobs_per_run: 15
  job_types:
    - "ml_training"
    - "data_analytics"
    - "simulation"
    - "genomics"

fault_injection:
  enabled: true
  fault_rate: 0.2
  fault_types:
    - "crash"
    - "byzantine"
    - "network_partition"
    - "slow_response"

evaluation:
  repetitions: 5
  parallel_execution: true
  timeout_minutes: 30

output:
  results_dir: "results"
  generate_plots: true
  detailed_logs: true
```

**Usage with config:**
```bash
python run_experimental_campaign.py --config example_config.yaml
```

## Evaluation Scenarios

### **Recommended: 5-Phase Experimental Campaign**

Following the comprehensive EXPERIMENTS.md design:

```bash
# Complete research-grade evaluation (1620 experiments)
python llm_experimental_framework.py --full-campaign

# This automatically runs all 5 phases:
# Phase 1: Baseline (120 experiments × 5 reps = 600 runs)
# Phase 2: LLM Evaluation (240 experiments × 3 reps = 720 runs) 
# Phase 3: Specialization (120 experiments × 5 reps = 600 runs)
# Phase 4: Scalability (96 experiments × 2 reps = 192 runs)
# Phase 5: Validation (48 experiments × 10 reps = 480 runs)
```

### **Quick Evaluation Scenarios**

#### 1. Baseline Performance Testing
```bash
# Using new framework (recommended)
python llm_experimental_framework.py --quick

# Using legacy framework  
python run_experimental_campaign.py \
  --protocols "byzantine_fault_tolerant,raft" \
  --agents 5 --jobs 20 --runs 5 \
  --fault-rate 0.0
```

#### 2. LLM vs Heuristic Comparison  
```bash
# Comprehensive LLM vs Heuristic analysis (recommended)
python llm_experimental_framework.py --llm-vs-heuristic

# Legacy approach with separate runs
python run_experimental_campaign.py --agent-mode heuristic --output-dir "results/heuristic"
python run_experimental_campaign.py --agent-mode llm --output-dir "results/llm"
python run_experimental_campaign.py --agent-mode hybrid --output-dir "results/hybrid"
```

#### 3. Fault Tolerance Evaluation
```bash
# Legacy framework fault injection
python run_experimental_campaign.py \
  --protocols "byzantine_fault_tolerant,raft" \
  --agents 7 --jobs 15 --runs 10 \
  --fault-rate 0.3
```

#### 4. Scalability Testing  
```bash
# Legacy framework scalability sweep
for agents in 3 5 7 10 15; do
  python run_experimental_campaign.py \
    --agents $agents --jobs 10 --runs 3 \
    --output-dir "results/scalability_$agents"
done
```

**Note**: The new `llm_experimental_framework.py` includes comprehensive fault injection, scalability testing, and LLM comparison as part of the 5-phase design.

## Results and Analysis

### **5-Phase Campaign Output Structure**

The `llm_experimental_framework.py` creates comprehensive results following EXPERIMENTS.md:

```
experimental_results_TIMESTAMP/
├── campaign_config.yaml           # Complete campaign configuration
├── phase_summaries/               # Phase-specific analysis
│   ├── phase_1_baseline_report.json
│   ├── phase_2_llm_evaluation_report.json
│   ├── phase_3_specialization_report.json
│   ├── phase_4_scalability_report.json
│   └── phase_5_validation_report.json
├── research_questions/            # RQ-specific analysis
│   ├── rq1_llm_vs_heuristic_analysis.json
│   ├── rq2_protocol_effectiveness.json
│   ├── rq3_specialization_impact.json
│   └── rq4_scalability_resilience.json
├── statistical_analysis/         # Comprehensive statistical tests
│   ├── anova_results.json
│   ├── t_test_comparisons.json
│   ├── effect_sizes.json
│   └── confidence_intervals.json
├── visualizations/               # Publication-ready plots
│   ├── llm_vs_heuristic_comparison.png
│   ├── protocol_effectiveness_radar.png
│   ├── specialization_benefits.png
│   ├── scalability_curves.png
│   └── fault_tolerance_heatmap.png
├── raw_data/                     # All experimental data
│   ├── phase_1/
│   ├── phase_2/
│   ├── phase_3/
│   ├── phase_4/
│   └── phase_5/
├── campaign_summary.md           # Executive summary
├── detailed_results.csv          # Combined dataset  
└── logs/
    ├── campaign_execution.log
    └── phase_logs/
```

### **Legacy Output Structure**

Each `run_experimental_campaign.py` evaluation creates:
```
results/
├── experiment_TIMESTAMP/
│   ├── config.yaml              # Experiment configuration
│   ├── raw_results.json         # Raw experimental data
│   ├── summary_report.md        # Human-readable summary
│   ├── statistical_analysis.json # Statistical test results
│   ├── plots/                   # Visualization files
│   │   ├── success_rates.png
│   │   ├── latency_comparison.png
│   │   └── fault_tolerance.png
│   └── logs/                    # Detailed execution logs
│       ├── experiment.log
│       └── agent_traces/
```

### Key Metrics

**Performance Metrics:**
- **Success Rate**: Percentage of jobs successfully allocated
- **Consensus Latency**: Time to reach consensus decisions
- **Resource Utilization**: Efficiency of resource allocation
- **Throughput**: Jobs processed per time unit

**Fault Tolerance Metrics:**
- **Recovery Time**: Time to recover from faults
- **Fault Detection Rate**: Percentage of faults detected
- **System Availability**: Uptime during fault conditions
- **Byzantine Resilience**: Tolerance to malicious agents

**LLM-Specific Metrics:**
- **Decision Quality**: Accuracy of LLM-based decisions  
- **API Response Time**: LLM query latency
- **Fallback Rate**: Frequency of fallback to heuristics
- **Token Usage**: LLM computational costs

## Advanced Usage

### Custom Fault Scenarios

Create custom fault injection patterns:

```python
# Custom fault configuration
custom_faults = {
    "coordinated_attack": {
        "type": "byzantine",
        "affected_agents": ["agent_1", "agent_2"],
        "duration": 300,  # seconds
        "behavior": "coordinated_malicious"
    },
    "network_split": {
        "type": "partition", 
        "partition_groups": [["agent_1", "agent_2"], ["agent_3", "agent_4", "agent_5"]],
        "duration": 180
    }
}
```

### Performance Profiling

Enable detailed performance profiling:
```bash
python -m cProfile -o profile.stats run_experimental_campaign.py \
  --protocols "raft" --agents 5 --jobs 10 --verbose
```

### Distributed Evaluation

For large-scale evaluations, distribute across multiple machines:
```bash
# Coordinator node
python run_experimental_campaign.py \
  --coordinator --workers worker1:8080,worker2:8080

# Worker nodes  
python run_experimental_campaign.py \
  --worker --port 8080
```

## Troubleshooting

### Common Issues

**1. SambaNova API Errors:**
```
Error: 403 Forbidden
Solution: Check API credentials and endpoint URL
```

**2. Memory Issues with Large Evaluations:**
```bash
# Reduce parallel runs or agent count
python run_experimental_campaign.py --agents 3 --jobs 5
```

**3. Timeout Issues:**
```bash
# Increase timeout for complex scenarios
python run_experimental_campaign.py --timeout-minutes 60
```

### Debug Mode

Enable comprehensive debugging:
```bash
python run_experimental_campaign.py \
  --debug --log-level DEBUG \
  --detailed-traces
```

## Extending the Evaluation Framework

### Adding New Protocols

1. Implement protocol class in `consensus_protocols/`
2. Add to protocol registry in `run_experimental_campaign.py`
3. Update configuration schema

### Custom Metrics

Add domain-specific metrics:
```python
def custom_metric_calculator(results):
    # Custom metric calculation
    return metric_value
```

### Integration with CI/CD

Automated evaluation in continuous integration:
```yaml
# .github/workflows/evaluation.yml
name: Consensus Evaluation
on: [push, pull_request]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Evaluation
        run: python evaluation/run_experimental_campaign.py --ci-mode
```

## Research Applications

### Academic Papers

The evaluation framework supports research for:
- **Distributed Systems Conferences** (SOSP, OSDI, NSDI)
- **AI/ML Conferences** (NeurIPS, ICML, ICLR) for LLM integration
- **HPC Conferences** (SC, HPDC, IPDPS) for resource scheduling

### Experimental Design

Follow these guidelines for rigorous evaluation:

1. **Statistical Power**: Use sufficient repetitions (≥10 runs)
2. **Control Variables**: Fix non-test parameters across experiments
3. **Randomization**: Use different random seeds for each run
4. **Baseline Comparison**: Include standard algorithms as baselines

### Publication Ready Results

Generate publication-quality outputs:
```bash
python run_experimental_campaign.py \
  --publication-mode \
  --latex-tables \
  --high-res-plots \
  --statistical-tests
```

## Performance Benchmarks

### Expected Performance

On standard hardware (8-core CPU, 16GB RAM):

| Protocol | Agents | Jobs | Avg Time | Success Rate |
|----------|--------|------|----------|--------------|
| BFT      | 5      | 10   | 45s      | 95%          |
| Raft     | 5      | 10   | 30s      | 98%          |
| Weighted | 5      | 10   | 25s      | 92%          |

### Scaling Characteristics

- **Agent Count**: Linear scaling up to ~20 agents
- **Job Count**: Sub-linear scaling (consensus overhead)
- **Fault Rate**: Exponential impact on latency above 30%

## Support and Development

### Contributing

1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Submit pull request with evaluation results

### Issues and Bugs

Report issues with:
- Experiment configuration
- Expected vs actual results  
- System logs and error messages
- Performance characteristics

### Future Enhancements

Planned improvements:
- **Real-time monitoring** dashboard
- **Machine learning** for parameter optimization
- **Cloud deployment** automation
- **Interactive result** exploration

---

For additional help or questions about the evaluation framework, please refer to the project documentation or open an issue in the repository.

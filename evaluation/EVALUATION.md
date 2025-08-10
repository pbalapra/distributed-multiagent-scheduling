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
   pip install pyyaml numpy matplotlib seaborn pandas
   ```

### Basic Evaluation Run

```bash
cd evaluation
python run_experimental_campaign.py
```

This runs a basic evaluation campaign with default settings.

## Evaluation Scripts

### `run_experimental_campaign.py`

The main experimental campaign runner that orchestrates comprehensive evaluations.

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

### 1. Baseline Performance Testing
Tests consensus protocols under normal operating conditions:
```bash
python run_experimental_campaign.py \
  --protocols "byzantine_fault_tolerant,raft" \
  --agents 5 --jobs 20 --runs 5 \
  --fault-rate 0.0
```

### 2. Fault Tolerance Evaluation
Injects various fault types to test resilience:
```bash
python run_experimental_campaign.py \
  --protocols "byzantine_fault_tolerant,raft" \
  --agents 7 --jobs 15 --runs 10 \
  --fault-rate 0.3
```

### 3. Scalability Testing
Tests performance with increasing agent counts:
```bash
for agents in 3 5 7 10 15; do
  python run_experimental_campaign.py \
    --agents $agents --jobs 10 --runs 3 \
    --output-dir "results/scalability_$agents"
done
```

### 4. LLM vs Heuristic Comparison
Compares AI-enhanced vs traditional decision making:
```bash
# Heuristic agents
python run_experimental_campaign.py \
  --agent-mode heuristic \
  --output-dir "results/heuristic"

# LLM agents  
python run_experimental_campaign.py \
  --agent-mode llm \
  --output-dir "results/llm"

# Hybrid agents
python run_experimental_campaign.py \
  --agent-mode hybrid \
  --output-dir "results/hybrid"
```

## Results and Analysis

### Output Structure

Each evaluation run creates:
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

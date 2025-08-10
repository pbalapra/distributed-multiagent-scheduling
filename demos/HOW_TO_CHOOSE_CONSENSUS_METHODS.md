# How to Choose Consensus Methods During Experiments

You now have a flexible experimental framework that allows you to select and configure different consensus methods at runtime. Here's your complete guide:

## 🚀 Quick Start Examples

### 1. Command Line Selection
```bash
# Test specific methods
python consensus_experiment_runner.py --methods pbft tendermint --agents 7 --jobs 5

# Test all available methods
python consensus_experiment_runner.py --methods all --repetitions 3

# Compare Byzantine fault tolerant methods only
python consensus_experiment_runner.py --methods pbft tendermint bft

# Compare crash fault tolerant methods
python consensus_experiment_runner.py --methods multi_paxos raft

# Quick test with reduced output
python consensus_experiment_runner.py --methods pbft tendermint --jobs 3 --repetitions 1 --quiet
```

### 2. Configuration File Selection
```bash
# Use pre-defined configurations
python consensus_experiment_runner.py --config quick_test.yaml
python consensus_experiment_runner.py --config comprehensive.yaml
python consensus_experiment_runner.py --config byzantine_focus.yaml
python consensus_experiment_runner.py --config crash_tolerance.yaml
```

## 📁 Available Consensus Methods

| Method | Code | Fault Model | Best For |
|--------|------|-------------|----------|
| **PBFT** | `pbft` | Byzantine | Security-critical systems |
| **Tendermint** | `tendermint` | Byzantine + Finality | Real-time allocation |
| **Multi-Paxos** | `multi_paxos` | Crash | Proven reliability |
| **BFT** | `bft` | Byzantine | Basic Byzantine tolerance |
| **Raft** | `raft` | Crash | Leader-based systems |
| **Negotiation** | `negotiation` | Agreement | Multi-round discussions |
| **Weighted Voting** | `weighted_voting` | Expertise | Specialist-driven decisions |
| **Bidding** | `bidding` | Economic | Resource competition |

## ⚙️ Configuration Options

### Sample Configuration Files

#### 1. `quick_test.yaml` - Fast Testing
```yaml
methods: ["pbft", "tendermint"]
num_agents: 7
num_jobs: 3
repetitions: 1
detailed_logging: true
```

#### 2. `comprehensive.yaml` - Complete Study
```yaml
methods: ["all"]
num_agents: 9
num_jobs: 10
repetitions: 5
byzantine_faults: 2
detailed_logging: false
```

#### 3. `byzantine_focus.yaml` - Byzantine Fault Testing
```yaml
methods: ["pbft", "tendermint", "bft"]
num_agents: 7
byzantine_faults: 2
num_jobs: 8
repetitions: 3
```

#### 4. `crash_tolerance.yaml` - Crash Fault Testing
```yaml
methods: ["multi_paxos", "raft"]
num_agents: 5
crash_faults: 2
num_jobs: 6
repetitions: 4
```

## 🔬 Experimental Scenarios

### For Your Technical Paper Research:

#### 1. **Performance Comparison Study**
```bash
python consensus_experiment_runner.py \
  --methods pbft tendermint multi_paxos \
  --agents 7 --jobs 10 --repetitions 5 \
  --output-dir performance_study
```

#### 2. **Scalability Analysis**
```bash
# Small scale (5 agents)
python consensus_experiment_runner.py --methods all --agents 5 --jobs 8 --repetitions 3

# Medium scale (7 agents)
python consensus_experiment_runner.py --methods all --agents 7 --jobs 10 --repetitions 3

# Large scale (9 agents)
python consensus_experiment_runner.py --methods all --agents 9 --jobs 12 --repetitions 3
```

#### 3. **Fault Tolerance Evaluation**
```bash
# No faults baseline
python consensus_experiment_runner.py --methods pbft tendermint --no-faults

# With Byzantine faults
python consensus_experiment_runner.py --methods pbft tendermint --byzantine-faults 2

# Heavy fault scenario
python consensus_experiment_runner.py --methods pbft tendermint --byzantine-faults 3 --agents 10
```

#### 4. **Message Efficiency Study**
```bash
python consensus_experiment_runner.py \
  --methods pbft multi_paxos tendermint \
  --jobs 15 --repetitions 10 \
  --output-dir message_efficiency
```

## 🛠️ Custom Configuration Creation

### Create Your Own Config File
```yaml
# my_experiment.yaml
methods: ["pbft", "tendermint"]           # Choose specific methods
num_agents: 8                             # Number of agents
byzantine_faults: 2                       # Byzantine faults to inject
num_jobs: 12                              # Number of test jobs
repetitions: 5                            # Statistical significance

# Job generation parameters
job_types: ["ai", "climate", "physics"]   # Types of HPC jobs
nodes_per_agent: [15, 25]                # Nodes per agent range
cpu_range: [32, 256]                     # CPU cores per node
memory_range: [128, 1024]                # Memory GB per node
gpu_range: [0, 12]                       # GPUs per node

# Experiment control
timeout_seconds: 60                       # Timeout per consensus
enable_faults: true                       # Enable fault injection
detailed_logging: true                    # Verbose output

# Output options
output_dir: "my_experiment_results"       # Results directory
save_raw_data: true                       # Save detailed results
```

Then run: `python consensus_experiment_runner.py --config my_experiment.yaml`

## 📊 Results and Analysis

### Automatic Outputs
- **Success rates** for each method
- **Average response times** 
- **Message efficiency** (messages per consensus)
- **Fault tolerance** demonstrations
- **Statistical rankings**

### Saved Files
- `experiment_summary.json` - Aggregated statistics
- `experiment_config.json` - Configuration used
- `raw_results.json` - Detailed per-job results

## 🎯 Choosing the Right Method for Your Use Case

### For **Security-Critical** HPC Systems:
```bash
python consensus_experiment_runner.py --methods pbft tendermint --byzantine-faults 2
```

### For **High-Performance** Requirements:
```bash
python consensus_experiment_runner.py --methods tendermint multi_paxos --repetitions 10
```

### For **Proven Reliability**:
```bash
python consensus_experiment_runner.py --methods multi_paxos raft --repetitions 5
```

### For **Research Comparison** (All Methods):
```bash
python consensus_experiment_runner.py --methods all --repetitions 5 --agents 9 --jobs 15
```

## 🔄 Dynamic Method Selection During Runtime

### Interactive Selection (Future Enhancement)
The framework can be extended to support interactive method selection:

```python
# Example: Interactive mode (not yet implemented)
python consensus_experiment_runner.py --interactive
# Would prompt: "Select consensus methods: 1) PBFT 2) Tendermint 3) Multi-Paxos..."
```

### Conditional Method Selection
```yaml
# config with conditions
methods: 
  - name: "pbft"
    condition: "byzantine_faults > 0"
  - name: "multi_paxos"
    condition: "crash_faults > 0"
  - name: "tendermint"
    condition: "always"
```

## 🚀 Advanced Usage Examples

### Batch Experiments for Paper Results
```bash
# Run multiple experiment configurations
for config in byzantine_focus crash_tolerance comprehensive; do
  python consensus_experiment_runner.py --config $config.yaml --output-dir results_$config
done
```

### Parameter Sweep
```bash
# Test different agent counts
for agents in 5 7 9; do
  python consensus_experiment_runner.py --methods pbft tendermint --agents $agents --output-dir results_agents_$agents
done
```

### Fault Injection Study
```bash
# Test increasing Byzantine faults
for faults in 0 1 2 3; do
  python consensus_experiment_runner.py --methods pbft --byzantine-faults $faults --output-dir results_faults_$faults
done
```

## 🎯 Summary

You can now:

1. **Command Line**: Choose methods with `--methods pbft tendermint multi_paxos`
2. **Config Files**: Create YAML configurations for complex experiments
3. **Sample Configs**: Use pre-made configs for common scenarios
4. **Flexible Parameters**: Adjust agents, jobs, repetitions, faults
5. **Batch Processing**: Run multiple configurations automatically
6. **Statistical Analysis**: Get comprehensive performance comparisons

This framework gives you complete control over consensus method selection and experimental design for your technical paper research!

# How to Parameterize Fault Injection in Consensus Experiments

This guide shows you how to configure and parameterize sophisticated fault injection for your multi-agent consensus experiments.

## 🔥 **Fault Types Available**

### **1. Byzantine Faults (Malicious Behavior)**
- **Type**: `byzantine`
- **Impact**: Agents behave maliciously, send conflicting messages
- **Parameters**:
  - `byzantine_strategy`: `"random"`, `"selective"`, `"coordinated"`
  - `malicious_vote_rate`: Rate of malicious voting (0.0-1.0)

### **2. Crash Faults (Complete Failure)**
- **Type**: `crash`
- **Impact**: Agents become completely unavailable
- **Parameters**: Duration, recovery settings

### **3. Network Partitions**
- **Type**: `partition`
- **Impact**: Network splits into isolated groups
- **Parameters**:
  - `partition_groups`: Custom partition layout
  - Auto-generates random partitions if not specified

### **4. Message Delays**
- **Type**: `delay`
- **Impact**: Communication latency issues
- **Parameters**:
  - `delay_range`: `[min_seconds, max_seconds]`

### **5. Message Corruption**
- **Type**: `corruption`
- **Impact**: Data integrity issues in messages
- **Parameters**:
  - `corruption_rate`: Probability of message corruption (0.0-1.0)

### **6. Partial Failures**
- **Type**: `partial`
- **Impact**: Performance degradation
- **Parameters**:
  - `performance_degradation`: Performance factor (0.0-1.0)

### **7. Resource Exhaustion**
- **Type**: `exhaustion`
- **Impact**: Limited resource availability
- **Parameters**:
  - `resource_limit`: Available resources fraction (0.0-1.0)

### **8. Intermittent Faults**
- **Type**: `intermittent`
- **Impact**: On/off failure patterns
- **Parameters**: On/off durations

### **9. Cascading Failures**
- **Type**: `cascade`
- **Impact**: Failures spread to other agents
- **Parameters**: Cascade probability

### **10. Timing Attacks**
- **Type**: `timing`
- **Impact**: Selective timing manipulation
- **Parameters**: Delay factors

## ⚙️ **Parameterization Methods**

### **Method 1: Command Line Parameters**

```bash
# Basic fault intensity
python enhanced_consensus_experiment_runner.py \
  --methods pbft tendermint \
  --fault-intensity heavy \
  --fault-target-fraction 0.4

# Specific fault scenarios
python enhanced_consensus_experiment_runner.py \
  --fault-scenarios network_chaos heavy_byzantine \
  --agents 9 --jobs 10

# No faults (baseline)
python enhanced_consensus_experiment_runner.py \
  --methods all --no-faults
```

### **Method 2: YAML Configuration Files**

#### **Basic Fault Configuration**
```yaml
# fault_basic.yaml
methods: ["pbft", "tendermint"]
num_agents: 7
num_jobs: 5
repetitions: 2

# Fault parameters
fault_scenarios: ["light_byzantine"]
fault_intensity: "medium"
fault_target_fraction: 0.3
fault_start_delay: 5.0
fault_duration_range: [10.0, 30.0]
recovery_enabled: true
```

#### **Advanced Fault Configuration**
```yaml
# fault_comprehensive.yaml
methods: ["pbft", "tendermint", "multi_paxos"]
num_agents: 9
num_jobs: 10
repetitions: 3

# Multiple fault scenarios
fault_scenarios: ["heavy_byzantine", "network_chaos"]
fault_intensity: "heavy"

# Targeting parameters
fault_target_fraction: 0.4
specific_fault_targets: ["AGENT_01", "AGENT_02"]

# Network fault parameters
message_delay_range: [0.5, 3.0]
message_corruption_rate: 0.15
network_partition_probability: 0.5

# Performance parameters
performance_degradation_factor: 0.3
resource_exhaustion_limit: 0.6
```

#### **Custom Fault Injection**
```yaml
# fault_custom.yaml
methods: ["pbft", "tendermint"]
num_agents: 8
num_jobs: 12

# Custom fault definitions
custom_faults:
  - fault_type: "byzantine"
    duration: 25.0
    target_fraction: 0.25
    severity: "high"
    byzantine_strategy: "coordinated"
    start_time: 10.0
  
  - fault_type: "delay"
    duration: 40.0
    target_agents: ["GPU_CLUSTER_MANAGER", "CPU_CLUSTER_MANAGER"]
    delay_range: [1.0, 5.0]
    start_time: 15.0
  
  - fault_type: "partial"
    duration: 60.0
    performance_degradation: 0.2
    target_fraction: 0.5
```

### **Method 3: Programmatic Configuration**

```python
from fault_injection_framework import FaultParameters, FaultType, FaultSeverity

# Create custom fault parameters
byzantine_fault = FaultParameters(
    fault_type=FaultType.BYZANTINE,
    probability=1.0,
    duration=30.0,
    severity=FaultSeverity.HIGH,
    target_fraction=0.3,
    byzantine_strategy="coordinated",
    start_time=8.0
)

network_delay = FaultParameters(
    fault_type=FaultType.MESSAGE_DELAY,
    probability=1.0,
    duration=45.0,
    target_fraction=0.5,
    delay_range=[0.5, 3.0],
    start_time=12.0
)

# Inject during experiment
fault_injector.inject_fault(byzantine_fault)
fault_injector.inject_fault(network_delay)
```

## 🎯 **Fault Intensity Levels**

### **Light Intensity**
```yaml
fault_intensity: "light"
# - 1 fault type
# - Low severity
# - 20% agent targeting
# - Easy recovery
```

### **Medium Intensity**  
```yaml
fault_intensity: "medium"
# - 2 fault types
# - Medium severity  
# - 30% agent targeting
# - Moderate recovery
```

### **Heavy Intensity**
```yaml
fault_intensity: "heavy"
# - 3 fault types
# - High severity
# - 40% agent targeting
# - Difficult recovery
```

### **Chaos Intensity**
```yaml
fault_intensity: "chaos"
# - 5+ fault types
# - Critical severity
# - 60% agent targeting
# - System-threatening
```

## 📋 **Predefined Fault Scenarios**

### **light_byzantine**
```yaml
fault_scenarios: ["light_byzantine"]
# Single Byzantine agent with low impact
```

### **heavy_byzantine**
```yaml
fault_scenarios: ["heavy_byzantine"] 
# Multiple coordinated Byzantine agents
```

### **network_chaos**
```yaml
fault_scenarios: ["network_chaos"]
# Network partitions + message delays + corruption
```

### **cascading_failure**
```yaml
fault_scenarios: ["cascading_failure"]
# Failures that spread to other agents
```

### **performance_degradation**
```yaml
fault_scenarios: ["performance_degradation"]
# Partial failures + resource exhaustion
```

### **mixed_chaos**
```yaml
fault_scenarios: ["mixed_chaos"]
# Byzantine + crash + delays + intermittent faults
```

## ⏰ **Temporal Fault Parameters**

### **Start Time Control**
```yaml
fault_start_delay: 10.0        # Start faults after 10 seconds
fault_duration_range: [15, 45] # Random duration between 15-45s
```

### **Recovery Settings**
```yaml
recovery_enabled: true
recovery_time: 8.0             # Time to recover after fault ends
```

### **Custom Timing**
```yaml
custom_faults:
  - fault_type: "byzantine"
    start_time: 5.0            # Start at 5 seconds
    end_time: 35.0             # End at 35 seconds (overrides duration)
    recovery_time: 10.0        # Custom recovery time
```

## 🎯 **Agent Targeting Parameters**

### **Fraction-Based Targeting**
```yaml
fault_target_fraction: 0.4    # Target 40% of agents randomly
```

### **Specific Agent Targeting** 
```yaml
specific_fault_targets:        # Target specific agents
  - "PRIMARY_CONTROLLER"
  - "GPU_CLUSTER_MANAGER"
  - "MEMORY_MANAGER"
```

### **Mixed Targeting**
```yaml
custom_faults:
  - fault_type: "crash"
    target_agents: ["BACKUP_CONTROLLER"]  # Specific targeting
  - fault_type: "delay" 
    target_fraction: 0.5                  # Fraction targeting
```

## 📡 **Network Fault Parameters**

### **Message Delays**
```yaml
message_delay_range: [0.2, 4.0]  # 200ms to 4 seconds delay
```

### **Message Corruption**
```yaml
message_corruption_rate: 0.12   # 12% of messages corrupted
```

### **Network Partitions**
```yaml
network_partition_probability: 0.3
partition_groups:               # Custom partition layout
  - ["AGENT_01", "AGENT_02", "AGENT_03"]
  - ["AGENT_04", "AGENT_05"] 
  - ["AGENT_06", "AGENT_07"]
```

## 📊 **Performance Fault Parameters**

### **Performance Degradation**
```yaml
performance_degradation_factor: 0.25  # 25% performance (75% degradation)
```

### **Resource Limits**
```yaml
resource_exhaustion_limit: 0.4        # 40% resources available
```

## 🚀 **Running Fault-Injected Experiments**

### **Basic Command Line**
```bash
# Medium intensity with network chaos
python enhanced_consensus_experiment_runner.py \
  --methods pbft tendermint \
  --fault-scenarios network_chaos \
  --fault-intensity medium

# Heavy Byzantine testing
python enhanced_consensus_experiment_runner.py \
  --methods all \
  --fault-scenarios heavy_byzantine cascading_failure \
  --fault-target-fraction 0.5 \
  --repetitions 3
```

### **Configuration File Usage**
```bash
# Use predefined configurations
python enhanced_consensus_experiment_runner.py --config fault_basic.yaml
python enhanced_consensus_experiment_runner.py --config fault_comprehensive.yaml
python enhanced_consensus_experiment_runner.py --config fault_chaos.yaml
python enhanced_consensus_experiment_runner.py --config fault_network.yaml
```

### **Parameter Sweep Studies**
```bash
# Test different fault intensities
for intensity in light medium heavy chaos; do
  python enhanced_consensus_experiment_runner.py \
    --methods pbft tendermint \
    --fault-intensity $intensity \
    --output-dir results_$intensity
done

# Test different target fractions
for fraction in 0.2 0.4 0.6; do
  python enhanced_consensus_experiment_runner.py \
    --fault-scenarios mixed_chaos \
    --fault-target-fraction $fraction \
    --output-dir results_fraction_$fraction
done
```

## 📈 **Fault Impact Analysis**

The system automatically generates fault-aware analysis including:

### **Fault Resistance Metrics**
- Success rate under faults
- Fault resistance percentage  
- Recovery time analysis
- Message efficiency under faults

### **Agent Status Tracking**
- Real-time availability status
- Performance degradation factors
- Active fault counts per agent
- Byzantine behavior detection

### **Comprehensive Logs**
- Fault injection timeline
- Fault activation/deactivation events
- Agent state transitions
- Recovery success/failure rates

## 🎯 **Best Practices for Fault Parameterization**

### **1. Baseline Testing**
```bash
# Always start with no faults
python enhanced_consensus_experiment_runner.py --no-faults
```

### **2. Gradual Intensity Increase**
```bash
# Start light, increase gradually
--fault-intensity light   # First test
--fault-intensity medium  # Second test  
--fault-intensity heavy   # Third test
```

### **3. Single Fault Type Testing**
```yaml
# Test one fault type at a time first
fault_scenarios: ["light_byzantine"]  # Byzantine only
fault_scenarios: ["network_chaos"]    # Network only
```

### **4. Realistic Parameter Ranges**
```yaml
# Use realistic parameters for your environment
message_delay_range: [0.1, 2.0]      # Typical network delays
performance_degradation_factor: 0.7   # Realistic performance drop
fault_target_fraction: 0.3            # Reasonable fault coverage
```

### **5. Statistical Significance**
```yaml
repetitions: 5  # Minimum for statistical significance
repetitions: 10 # Better confidence intervals
```

## 🏁 **Summary**

You can now parameterize fault injection through:

1. **🖥️ Command Line**: Quick parameter adjustment
2. **📄 YAML Files**: Complex scenario definitions  
3. **🔧 Programmatic**: Full custom control
4. **📊 Intensity Levels**: Predefined severity scales
5. **⏰ Temporal Control**: Precise timing management
6. **🎯 Agent Targeting**: Flexible targeting strategies
7. **📡 Network Parameters**: Communication fault control
8. **📊 Performance Tuning**: Resource limitation settings

This comprehensive fault injection framework allows you to test consensus protocols under realistic failure conditions with complete parameter control for your technical paper research!

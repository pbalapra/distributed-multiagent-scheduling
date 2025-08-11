# Distributed Multi-Agent Scheduling for Resilient HPC

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxxx-blue)](https://doi.org/10.xxxx/xxxxxx)

A comprehensive implementation and evaluation framework for distributed multi-agent job scheduling in high-performance computing (HPC) environments. This repository contains the complete codebase for the paper **"Distributed Multi-Agent Scheduling for Resilient High-Performance Computing: Experimental Evaluation"**.

## 🎯 Method Overview

### **Core Concept**
The system implements a **decentralized scheduling architecture** where multiple autonomous agents collaborate to schedule jobs across distributed computing resources, replacing traditional centralized schedulers that create single points of failure.

### **Key Components**

**1. Multi-Agent Architecture**
- **Resource Agents**: Autonomous agents managing individual compute nodes/clusters
- **Distributed Coordination**: No central scheduler - agents negotiate directly
- **Competitive Bidding**: Agents bid for jobs based on resource availability and capability matching

**2. Event-Driven Scheduling**
- **Discrete Event Simulation**: Pure event-driven approach (no polling)
- **Priority Queue Management**: O(log n) complexity for scalable event processing
- **Message-Passing Protocol**: Asynchronous communication between agents

**3. Fault Tolerance Mechanisms**
- **Heartbeat Monitoring**: Continuous health checking of all agents
- **Automatic Recovery**: Failed jobs automatically redistributed
- **No Single Point of Failure**: System continues operating even if multiple agents fail

### **Scheduling Algorithm**

**Phase 1: Job Arrival**
```
1. Job submitted to system
2. Resource agents evaluate job requirements
3. Capable agents generate competitive bids
4. Bids include resource availability scores
```

**Phase 2: Competitive Selection**
```
1. Agents compete based on multi-factor scoring:
   - CPU/Memory/GPU availability match
   - Current workload and utilization  
   - Historical performance metrics
2. Best-fit agent selected automatically
3. Job assignment and execution initiated
```

**Phase 3: Fault Handling**
```
1. Continuous monitoring via heartbeat protocol
2. Failure detection triggers automatic recovery
3. Failed jobs redistributed to available agents
4. System maintains >95% availability under failures
```

### **Technical Innovations**

**1. Scalable Event Processing**
- Heap-based priority queue for O(log n) event scheduling
- No time-driven polling - purely reactive system
- Efficient message routing and processing

**2. Intelligent Resource Matching**
- Multi-dimensional scoring algorithm considering CPU, memory, GPU requirements
- Dynamic capability assessment and load balancing
- Preference-based job placement optimization

**3. Resilience Through Redundancy**
- Distributed state management across multiple agents
- Automatic job retry and rescheduling mechanisms
- Cascading failure prevention through isolation

### **Advantages Over Centralized Approaches**

1. **Eliminates Single Point of Failure**: No central scheduler to fail
2. **Better Fault Tolerance**: System degrades gracefully under failures
3. **Improved Scalability**: Distributed decision-making reduces bottlenecks
4. **Adaptive Resource Management**: Agents respond dynamically to changing conditions
5. **Lower Latency**: Direct agent-to-agent communication reduces delays

### **Research Contributions**

- **Novel distributed coordination protocol** for HPC job scheduling
- **Competitive bidding mechanism** optimizing resource utilization
- **Comprehensive fault tolerance framework** with automatic recovery
- **Scalable event-driven architecture** supporting large-scale deployments

This approach represents a paradigm shift from traditional centralized HPC schedulers to resilient, self-organizing distributed systems that maintain performance even under significant infrastructure failures.

## 🚀 Key Features

- **Distributed Multi-Agent Architecture**: Autonomous agents with competitive bidding and fault tolerance
- **Discrete Event Simulation**: High-performance event-driven scheduling simulation
- **Comprehensive Evaluation Framework**: 26 test configurations across 5 experimental dimensions
- **Fault Injection & Recovery**: Configurable failure patterns and autonomous recovery mechanisms

## 🏗️ Architecture Overview

```
multiagent/
├── src/                          # Core implementation
│   ├── agents/                   # Multi-agent system
│   │   ├── base_agent.py        # Base agent with heartbeat monitoring
│   │   ├── resource_agent.py    # Resource management and job execution
│   │   └── llm_resource_agent.py # LLM-enhanced agent implementation
│   ├── communication/           # Message passing infrastructure
│   │   └── protocol.py          # Pub-sub messaging with fault tolerance
│   ├── scheduler/               # Scheduling algorithms
│   │   ├── discrete_event_scheduler.py  # Event-driven coordination
│   │   ├── llm_enhanced_scheduler.py    # LLM-powered scheduling
│   │   ├── scheduler.py         # Base scheduler interface
│   │   └── job_pool.py          # Job queue management
│   ├── jobs/                    # Job management
│   │   └── job.py              # Job lifecycle and dependencies
│   ├── resources/              # Resource modeling
│   │   └── resource.py         # HPC resource abstraction
│   └── llm/                    # LLM integration
│       ├── llm_interface.py    # LLM service interface
│       ├── context_manager.py  # LLM context management
│       └── ollama_provider.py  # Ollama LLM provider
├── evaluation/                  # Evaluation framework
│   ├── systematic_resilience_evaluation.py  # Main evaluation suite
│   ├── run_experimental_campaign.py         # Comprehensive experiments
│   ├── fault_tolerant_test.py   # Fault tolerance testing
│   ├── high_throughput_test.py  # Performance benchmarking
│   └── EVALUATION.md            # Evaluation documentation
├── demos/                       # Interactive demonstrations
│   ├── hybrid_llm_demo.py      # LLM fault tolerance demo
│   └── DEMO.md                 # Demo documentation
├── archive/                     # Archived materials
│   ├── deprecated_scripts/     # Legacy evaluation scripts
│   ├── publication_materials/  # Research figures and reports
│   └── old_demos/             # Previous demonstration versions
└── main.py                     # Unified CLI entry point
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Quick Install

```bash
# Clone the repository
git clone https://github.com/username/distributed-multiagent-scheduling.git
cd distributed-multiagent-scheduling

# Create virtual environment (recommended)
python -m venv multi-agent
source multi-agent/bin/activate  # On Windows: multi-agent\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dependencies

Core dependencies:
- `numpy >= 1.21.0` - Numerical computations
- `matplotlib >= 3.5.0` - Visualization and figure generation  
- `pandas >= 1.4.0` - Data analysis and manipulation
- `seaborn >= 0.11.0` - Statistical visualization
- `dataclasses` - Data structure definitions (Python 3.7+)
- `langchain-community` - **Required** for SambaNova LLM integration
- `python-dotenv` - Environment variable management (.env file support)

### **Important: LLM Integration Setup**

**Environment Variable Setup:**
```bash
# Option 1: Add to ~/.bashrc (permanent)
echo 'export SAMBASTUDIO_URL=your_sambanova_endpoint' >> ~/.bashrc
echo 'export SAMBASTUDIO_API_KEY=your_api_key' >> ~/.bashrc
source ~/.bashrc

# Option 2: Create .env file (project-specific)
echo 'SAMBASTUDIO_URL=your_sambanova_endpoint' > .env
echo 'SAMBASTUDIO_API_KEY=your_api_key' >> .env

# Option 3: Get interactive help
python main.py --setup-env
```

## 🎆 New: Unified Command Interface

The system now features a **streamlined command-line interface** through `main.py` that provides easy access to all major features:

🔗 **Single Entry Point**: `python main.py` with multiple modes  
🔍 **Environment Validation**: `--check` flag verifies setup and dependencies  
🎯 **Interactive Demos**: `--interactive` launches full LLM fault tolerance demonstration  
🔬 **Research Evaluation**: `--evaluate` runs comprehensive experimental campaigns  
📚 **Educational**: Rich colored output with step-by-step guidance  


### **🔐 Environment Variables**

The system automatically loads environment variables from:
- `.env` file in the project directory
- `~/.env` file in your home directory  
- System environment variables

For **SambaNova LLM integration**, set:
```bash
SAMBASTUDIO_URL=your_sambanova_endpoint
SAMBASTUDIO_API_KEY=your_api_key
```

**Recommended setup:**
```bash
# Create .env file in project directory
echo 'SAMBASTUDIO_URL=your_endpoint' > .env
echo 'SAMBASTUDIO_API_KEY=your_key' >> .env

# Or get interactive help
python main.py --setup-env
```

### **⚠️ Troubleshooting Common Issues**

**1. "No module named 'langchain_community'" Error:**
```bash
# Reinstall requirements (should already be included)
pip install -r requirements.txt

# Verify installation
python -c "from langchain_community.llms.sambanova import SambaStudio; print('✅ Installation successful')"
```

**2. Environment Variables Not Loading:**
```bash
# If using .bashrc, ensure it's sourced
source ~/.bashrc

# Or run demos with explicit sourcing
source ~/.bashrc && python demos/hybrid_llm_demo.py

# Verify variables are loaded
echo $SAMBASTUDIO_URL
echo $SAMBASTUDIO_API_KEY
```

**3. LLM Demos Show "Fallback Mode":**
```bash
# Check environment variables
python main.py --check

# Set up credentials properly
python main.py --setup-env
```

**4. Permission Errors with Virtual Environment:**
```bash
# Ensure virtual environment is activated
source multi-agent/bin/activate  # Linux/Mac
# or
multi-agent\Scripts\activate     # Windows

# Then reinstall requirements
pip install -r requirements.txt
```

## 🚀 Quick Start

The system provides a unified command-line interface through `main.py`:

```bash
# 1. Check environment and setup
python main.py --check

# 2. Basic system overview and capabilities  
python main.py

# 3. Interactive LLM fault tolerance demo
python main.py --interactive

# 4. Full evaluation campaign
python main.py --evaluate

# Show all available options
python main.py --help
```

## 🧪 Experimental Framework

### Test Configurations

The evaluation framework includes 26 test configurations across 5 dimensions:

1. **Scale Testing** (12 configs): 50-500 jobs, 5-20 agents
2. **Failure Rate Testing** (4 configs): 5%-35% failure rates
3. **Failure Pattern Testing** (3 configs): Random, cascading, network partition
4. **Load Pattern Testing** (3 configs): Constant, burst, Poisson arrivals
5. **High Load Testing** (4 configs): 50-400 concurrent job bursts

### Evaluation Metrics

- **Job Completion Rate**: Primary success metric (%)
- **System Availability**: Operational uptime (%)
- **Fault Tolerance Score**: Composite resilience index (0-100)
- **Mean Time to Recovery**: Average failure recovery duration
- **Throughput**: Jobs completed per time unit

### Statistical Analysis

All results include:
- **Multiple repetitions** (3-5 per configuration)
- **Statistical significance testing** (p-values, effect sizes)
- **Confidence intervals** and variance analysis
- **Reproducible random seeds** for consistency

## 🛡️ Interactive Fault Tolerance Demo with LLM Integration

### **Hybrid LLM Fault Tolerance Demo**

This advanced demo showcases multi-agent fault-tolerant consensus with full **LLM prompt and response visibility**, Byzantine fault injection, and system recovery monitoring.

### **Key Features**

- **Real SambaNova LLM Integration**: Uses LangChain SambaStudio client for authentic LLM consensus
- **Full Prompt/Response Transparency**: Every LLM query and response is displayed for educational insight
- **Byzantine Fault Simulation**: Realistic malicious agent behavior with compromised LLM responses
- **Intelligent Fallback**: Graceful degradation to simulated responses if API unavailable
- **Visual Recovery Monitoring**: Real-time fault injection and recovery progress tracking
- **Consensus Protocol Analysis**: Detailed breakdown of voting, proposals, and consensus outcomes

### **Running the Demo**

```bash
# Option 1: Using main.py (recommended)
python main.py --interactive

# Option 2: Direct execution
python demos/hybrid_llm_demo.py
```

### **Demo Scenarios**

**Scenario 1: Healthy System Baseline**
- All 5 specialized agents (GPU, Memory, Compute, Storage, General) operate normally
- Demonstrates successful consensus on scientific job scheduling
- Shows full LLM prompt engineering and structured JSON responses

**Scenario 2: Byzantine Fault Injection**
- Selected agents become malicious and provide compromised proposals/votes
- System detects Byzantine behavior and rejects malicious consensus attempts
- Demonstrates 2/3 majority voting enforcement and fault tolerance

**Scenario 3: System Recovery**
- Faulty agents automatically recover after fault period
- System resumes normal consensus operations
- Shows resilience and self-healing capabilities

### **Sample Output**

```
🎯 DEMO: Multi-Agent Fault Tolerance with LLM Integration

🔧 Initializing 5 specialized agents with SambaNova LLM integration...
✅ All agents initialized successfully

📋 SCENARIO 1: Healthy System Baseline

🤖 GPU-Agent generating proposal for Climate-Modeling-Job...
💬 LLM QUERY [GPU-Agent-Proposal]:
```
You are a GPU specialist agent evaluating job placement. 
Job: Climate-Modeling-Job (Nodes: 4, CPU/node: 8, Memory/node: 32GB, GPUs: 2)
Available nodes: [detailed node specs...]
Generate a JSON proposal with job placement decision.
```

🤖 LLM RESPONSE:
```json
{
  "decision": "accept", 
  "assigned_nodes": ["n1", "n2", "n3", "n4"],
  "confidence": 0.92,
  "reasoning": "Excellent GPU match for climate modeling workload..."
}
```

🗳️ Consensus Result: ✅ ACCEPTED (4/5 votes)

⚠️ SCENARIO 2: Byzantine Fault Injection
[Agent-02] has been compromised (Byzantine fault)

🤖 Compromised agent generating malicious proposal...
💬 LLM RESPONSE (Malicious):
```json
{
  "decision": "accept",
  "assigned_nodes": ["invalid_node", "overloaded_node"],
  "confidence": 1.0,
  "reasoning": "Malicious placement to cause system failure"
}
```

🗳️ Consensus Result: ❌ REJECTED (2/5 votes) - Byzantine fault detected!

🔄 SCENARIO 3: System Recovery
✅ All agents recovered. Resuming normal operations...
🗳️ Consensus Result: ✅ ACCEPTED (5/5 votes)
```

### **Educational Value**

- **LLM Engineering**: See exactly how prompts are constructed for different agent specializations
- **Consensus Protocols**: Observe Byzantine fault tolerance in action with majority voting
- **Fault Injection**: Understand how malicious agents can compromise distributed systems
- **System Resilience**: Watch automatic recovery and consensus restoration
- **Practical Application**: Real-world HPC job scheduling with intelligent agent reasoning

### **Customization Options**

```python
# Modify demo parameters in hybrid_llm_demo.py
NUM_AGENTS = 7              # Increase agent count
NUM_JOBS = 3                # Adjust job scenarios
FAULT_DURATION = 30.0       # Fault injection period
RECOVERY_TIME = 10.0        # Recovery monitoring time

# LLM parameters
LLM_TEMPERATURE = 0.3       # Response determinism
MAX_TOKENS = 1000           # Response length limit
```

### **Environment Setup**

For full LLM functionality, configure SambaNova credentials:
```bash
# Check current setup
python main.py --check

# Interactive environment setup  
python main.py --setup-env
```

**Fallback Mode**: If SambaNova API is unavailable, the demo automatically uses simulated responses with full educational functionality.

## ⚙️ Configurable Experiment Runners

### **Consensus Experiment Runner**

A flexible framework for systematic consensus protocol evaluation with configurable parameters.

**Supported Consensus Methods:**
- **PBFT**: Practical Byzantine Fault Tolerance
- **Multi-Paxos**: Crash fault tolerant consensus
- **Tendermint**: Modern Byzantine consensus with finality
- **Raft**: Leader-based distributed consensus
- **Negotiation**: Multi-round iterative consensus
- **Weighted Voting**: Expertise-based decision making
- **Bidding**: Economic consensus mechanisms
- **Gossip**: Epidemic-style information propagation

**Agent Decision Modes:**
- **Heuristic**: Traditional rule-based agents
- **LLM**: Large language model enhanced agents
- **Hybrid**: Mixed pool of heuristic and LLM agents

### **Enhanced Experiment Runner with Fault Injection**

Advanced experiment runner integrating comprehensive fault injection capabilities.

**Fault Scenarios:**
- **Byzantine Faults**: Malicious agent behavior
- **Crash Faults**: Agent failures and restarts
- **Network Partitions**: Communication isolation
- **Message Delays**: Network latency simulation
- **Performance Degradation**: Resource exhaustion
- **Message Corruption**: Communication errors

**Fault Intensity Levels:**
- **Light**: 1 fault, 20% targets, low severity
- **Medium**: 2 faults, 30% targets, medium severity
- **Heavy**: 3 faults, 40% targets, high severity
- **Chaos**: 5 faults, 60% targets, critical severity

### **Configuration Options**

```yaml
# Example experiment configuration
methods: ["pbft", "tendermint", "multi_paxos"]
agent_decision_mode: "hybrid"  # "heuristic", "llm", "hybrid"
num_agents: 7
byzantine_faults: 2
crash_faults: 1
num_jobs: 5
job_types: ["ai", "climate", "genomics", "physics"]
repetitions: 3
timeout_seconds: 30
enable_faults: true

# Fault injection parameters
fault_scenarios: ["byzantine_cascade", "network_partition"]
fault_intensity: "medium"
fault_target_fraction: 0.3
fault_start_delay: 5.0
recovery_enabled: true
recovery_time: 5.0
```

### **Command Line Usage**

```bash
# Note: Consensus experiment runners not available in current version
```

### **Output Analysis**

**Standard Metrics:**
- Success rates and completion percentages
- Average consensus time and message counts
- Round complexity and convergence analysis
- Statistical significance testing

**Fault-Aware Metrics:**
- Fault resistance and recovery capabilities
- Impact analysis by fault type
- Agent availability during consensus
- Fault correlation with performance degradation

**Generated Files:**
```
experiment_results/
├── experiment_config.json       # Configuration snapshot
├── experiment_summary.json      # High-level results
├── raw_results.json            # Detailed per-job results
├── fault_aware_summary.json    # Fault injection analysis
├── fault_logs.json             # Detailed fault timeline
└── fault_results_per_method.json # Method-specific fault impact
```

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{distributed_multiagent_scheduling_2024,
  title={Fault-Tolerant Distributed Multi-Agent Scheduling for High-Performance Computing: A Resilience-Centric Approach},
  author={Prachi Jadhav, Fred Sutter, Ewa Deelman, Prasanna Balaprakash},
  journal={TBD},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Research conducted through SWARM project supported by the Department of Energy Award #DE-SC0024387.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/username/distributed-multiagent-scheduling/issues)
- **Discussions**: [GitHub Discussions](https://github.com/username/distributed-multiagent-scheduling/discussions)

---

**⭐ Star this repository if you find it useful for your research!**

# Distributed Multi-Agent Scheduling for Resilient HPC

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxxx-blue)](https://doi.org/10.xxxx/xxxxxx)

A comprehensive implementation and evaluation framework for distributed multi-agent job scheduling in high-performance computing (HPC) environments. This repository contains the complete codebase for the paper **"Distributed Multi-Agent Scheduling for Resilient High-Performance Computing: Experimental Evaluation"**.

## 🚀 Key Features

- **Distributed Multi-Agent Architecture**: Autonomous agents with competitive bidding and fault tolerance
- **Discrete Event Simulation**: High-performance event-driven scheduling simulation
- **Comprehensive Evaluation Framework**: 26 test configurations across 5 experimental dimensions
- **Fault Injection & Recovery**: Configurable failure patterns and autonomous recovery mechanisms
- **Publication-Ready Results**: Automated generation of research figures and statistical analysis
- **96.2% Win Rate**: Demonstrated superiority over centralized scheduling approaches

## 📊 Performance Highlights

- **25x better completion rate** under extreme load (400 concurrent jobs)
- **Graceful degradation**: Maintains 82% completion vs 28% centralized at 35% failure rates
- **Superior scalability**: 81-96% completion across varying workload sizes
- **Statistical significance**: p < 0.001, Cohen's d = 2.84 (large effect size)

## 🏗️ Architecture Overview

```
multiagent/
├── src/                          # Core implementation
│   ├── agents/                   # Multi-agent system
│   │   ├── base_agent.py        # Base agent with heartbeat monitoring
│   │   └── resource_agent.py    # Resource management and job execution
│   ├── communication/           # Message passing infrastructure
│   │   └── protocol.py          # Pub-sub messaging with fault tolerance
│   ├── scheduler/               # Scheduling algorithms
│   │   └── discrete_event_scheduler.py  # Event-driven coordination
│   ├── jobs/                    # Job management
│   │   └── job.py              # Job lifecycle and dependencies
│   └── resources/              # Resource modeling
│       └── resource.py         # HPC resource abstraction
├── evaluation/                  # Evaluation framework
│   ├── systematic_resilience_evaluation.py  # Main evaluation suite
│   ├── fault_tolerant_test.py   # Fault tolerance testing
│   └── high_throughput_test.py  # Performance benchmarking
├── demos/                       # Example implementations
├── figures/                     # Generated evaluation results
└── docs/                       # Documentation
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
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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

## 🚀 Quick Start

### Basic Usage

```python
from src.scheduler.discrete_event_scheduler import DiscreteEventScheduler
from src.agents.resource_agent import ResourceAgent
from src.resources.resource import Resource, ResourceCapacity, ResourceType

# Create scheduler and agents
scheduler = DiscreteEventScheduler()
agent = ResourceAgent("agent-1", resource, scheduler, failure_rate=0.1)

# Run simulation
scheduler.start()
# Submit jobs and monitor results
```

### Run Evaluation Suite

```bash
# Quick evaluation (2-3 minutes)
python evaluation/quick_resilience_test.py

# Ultra-quick demo (30 seconds)
python evaluation/ultra_quick_test.py

# Full systematic evaluation (30-45 minutes)
python evaluation/systematic_resilience_evaluation.py

# Generate publication figures
python create_bw_publication_figures.py
```

## 📈 Reproducing Paper Results

### Complete Evaluation Reproduction

```bash
# 1. Run systematic resilience evaluation
python evaluation/systematic_resilience_evaluation.py

# 2. Run fault tolerance tests  
python evaluation/fault_tolerant_test.py

# 3. Run high throughput benchmarks
python evaluation/high_throughput_test.py

# 4. Generate all publication figures
python create_bw_publication_figures.py

# 5. Compile LaTeX results document
pdflatex resilience_evaluation_results.tex
```

### Expected Runtime
- **Quick test**: 2-3 minutes (5 configurations)
- **Systematic evaluation**: 30-45 minutes (26 configurations) 
- **Complete reproduction**: 60-90 minutes (all tests + figures)

### Output Files
```
Generated Results:
├── bw_figures/                   # Black & white publication figures (11 files)
├── figures/                      # Color figures and statistical tables
├── resilience_study_results_*.json  # Raw evaluation data
└── resilience_evaluation_results.pdf  # LaTeX compiled results
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

## 📊 Key Results Summary

| Experimental Dimension | Configs | Distributed Wins | Win Rate | Avg Advantage |
|------------------------|---------|------------------|----------|---------------|
| Scale Testing          | 12      | 11               | 91.7%    | +52.3%        |
| Failure Rate Testing   | 4       | 4                | 100%     | +38.5%        |
| Failure Pattern Testing| 3       | 3                | 100%     | +55.7%        |
| Load Pattern Testing   | 3       | 3                | 100%     | +41.3%        |
| High Load Performance  | 4       | 4                | 100%     | +47.8%        |
| **Overall Results**    | **26**  | **25**           | **96.2%**| **+47.1%**    |

**Statistical Significance**: p < 0.001, Cohen's d = 2.84, Effect Size: Large

## 🔬 Advanced Usage

### Custom Evaluation Scenarios

```python
from evaluation.systematic_resilience_evaluation import ExperimentConfig, run_resilience_experiment

# Define custom experiment
config = ExperimentConfig(
    name="Custom-Test",
    num_jobs=100,
    num_agents=10,
    agent_failure_rate=0.2,
    scheduler_failure_rate=0.1,
    job_arrival_pattern='burst',
    failure_pattern='cascading',
    simulation_time=150.0,
    repetitions=5
)

# Run evaluation
results = run_resilience_experiment(config)
```

### Custom Agent Behavior

```python
from src.agents.resource_agent import ResourceAgent

class CustomAgent(ResourceAgent):
    def _calculate_job_score(self, job_data):
        # Custom scoring algorithm
        score = super()._calculate_job_score(job_data)
        # Add custom logic
        return modified_score
```

### Fault Injection Patterns

```python
from evaluation.systematic_resilience_evaluation import inject_failure_pattern

# Custom failure injection
def custom_failure_pattern(simulation, pattern, tracker, simulation_time):
    agents = list(simulation.agents.values())
    # Implement custom failure timing and patterns
    for agent in agents:
        agent.failure_time = custom_failure_schedule()
```

## 📚 Documentation

- **API Documentation**: See `docs/` directory
- **Architecture Guide**: `docs/ARCHITECTURE.md`
- **Evaluation Guide**: `docs/EVALUATION.md` 
- **Figure Descriptions**: `bw_figure_descriptions.md`
- **LaTeX Results**: `resilience_evaluation_results.tex`

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run evaluation validation
python tests/validate_evaluation.py
```

## 📈 Benchmarking

Performance benchmarks on standard hardware (Intel i7, 16GB RAM):

- **Simulation Rate**: ~10,000 events/second
- **Agent Scalability**: Up to 50 agents efficiently
- **Job Throughput**: 1,000+ jobs per simulation
- **Memory Usage**: <2GB for largest configurations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
python -m pytest
```

### Contribution Areas

- **New evaluation scenarios** and metrics
- **Performance optimizations** for large-scale simulations
- **Additional scheduling algorithms** for comparison
- **Visualization improvements** and interactive dashboards
- **Documentation** and tutorial improvements

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{distributed_multiagent_scheduling_2024,
  title={Distributed Multi-Agent Scheduling for Resilient High-Performance Computing: Experimental Evaluation},
  author={Authors},
  journal={Journal Name},
  year={2024},
  volume={XX},
  number={X},
  pages={XXX-XXX},
  doi={10.xxxx/xxxxxx}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Research supported by [Funding Agency]
- Computational resources provided by [Computing Center]
- Special thanks to the HPC scheduling research community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/username/distributed-multiagent-scheduling/issues)
- **Discussions**: [GitHub Discussions](https://github.com/username/distributed-multiagent-scheduling/discussions)
- **Email**: [authors@institution.edu](mailto:authors@institution.edu)

## 🔄 Version History

- **v1.0.0** (2024-01-XX): Initial release with complete evaluation framework
- **v0.9.0** (2023-12-XX): Beta release with systematic evaluation
- **v0.8.0** (2023-11-XX): Alpha release with basic multi-agent implementation

---

**⭐ Star this repository if you find it useful for your research!**
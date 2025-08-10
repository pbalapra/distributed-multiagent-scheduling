# Multi-Agent Fault Tolerance Demo

This directory contains the **Hybrid LLM Fault Tolerance Demo** - an educational demonstration of multi-agent Byzantine fault-tolerant consensus with Large Language Model integration.

## Overview

The `hybrid_llm_demo.py` script demonstrates a sophisticated multi-agent system that:

- **Multi-Agent Consensus**: Implements Byzantine Fault Tolerant (BFT) consensus among specialized HPC resource agents
- **LLM Integration**: Uses SambaNova's Meta-Llama models for intelligent decision-making
- **Fault Injection**: Simulates various failure modes including Byzantine attacks, crash failures, and network partitions
- **Educational Transparency**: Shows detailed prompts, responses, and decision-making processes
- **Graceful Fallbacks**: Handles API failures with realistic simulated responses

## Quick Start

### Prerequisites

1. **Environment Setup**:
   ```bash
   export SAMBASTUDIO_API_KEY="your-api-key-here"
   export SAMBASTUDIO_URL="your-sambastudio-url"
   ```

2. **Dependencies**:
   - Python 3.8+
   - LangChain SambaNova integration
   - Standard libraries: `asyncio`, `json`, `random`, `time`

### Running the Demo

```bash
cd /Users/p12/projects/2025/multiagent/demos
python hybrid_llm_demo.py
```

## What You'll See

The demo runs through several phases:

### Phase 1: System Initialization
- **Agent Setup**: 5 specialized agents (GPU, Memory, Compute, Storage, General)
- **Resource Allocation**: Each agent manages specific HPC resources
- **Network Configuration**: Peer-to-peer communication setup

### Phase 2: Healthy Operation
- **Job Submission**: Scientific computing jobs requiring HPC resources
- **Consensus Protocol**: Agents vote on job placement using LLM-driven decisions
- **Resource Management**: Dynamic allocation and release of computational resources

### Phase 3: Fault Injection
Multiple fault types are demonstrated:

- **Byzantine Faults**: Malicious agents providing false information
- **Crash Failures**: Agents becoming unresponsive
- **Network Partitions**: Communication failures between agents
- **Slow Responses**: Performance degradation simulation

### Phase 4: Recovery and Resilience
- **Fault Detection**: System identifies compromised agents
- **Consensus Under Attack**: BFT protocol maintains correctness despite faults
- **Agent Recovery**: Failed agents rejoin the system
- **Performance Analysis**: Success rates and efficiency metrics

## Key Features

### 🧠 LLM-Driven Decision Making

The demo showcases how Large Language Models can enhance multi-agent systems:

```python
# Example: Agent evaluating job placement
prompt = f"""
As a {agent_type} specialist managing HPC resources, evaluate this job:
Job: {job_name} requiring {cpu_cores} cores, {memory_gb}GB RAM
Current utilization: {utilization}%
Priority: {priority}

Provide your decision as JSON: {{"vote": "accept/reject", "confidence": 0.0-1.0, "reasoning": "..."}}
"""
```

### 🛡️ Byzantine Fault Tolerance

Implements a robust consensus protocol that:
- Tolerates up to ⌊(n-1)/3⌋ faulty agents
- Validates responses using cryptographic principles
- Maintains system integrity under attack

### 📊 Educational Transparency

Every step is logged with:
- **Detailed Prompts**: Exact text sent to LLMs
- **API Configurations**: Temperature, token limits, model parameters
- **Response Analysis**: JSON parsing, validation, fallback handling
- **Consensus Tracking**: Vote tallies, decision rationale, timeout handling

### 🔄 Graceful Degradation

The system handles various failure modes:
- **API Unavailability**: Falls back to rule-based decisions
- **Malformed Responses**: Intelligent parsing with error recovery
- **Network Issues**: Retry mechanisms with exponential backoff
- **Agent Failures**: Dynamic reconfiguration and load balancing

## Sample Output

```
🚀 HYBRID LLM FAULT TOLERANCE DEMO
==================================

📊 Initializing Multi-Agent HPC System...
✅ Agent GPU_SPECIALIST initialized - Managing 32 nodes with 8 GPUs each
✅ Agent MEMORY_SPECIALIST initialized - Managing 16 nodes with 512GB RAM each
✅ Agent COMPUTE_SPECIALIST initialized - Managing 64 nodes with 64 cores each
...

🔍 Phase 1: Healthy Operation
Submitting job: AI-Training-001
📤 PROMPT to GPU_SPECIALIST:
[Detailed prompt showing job requirements and decision criteria]

📥 RESPONSE from GPU_SPECIALIST:
{"vote": "accept", "confidence": 0.95, "reasoning": "Excellent fit for our GPU cluster..."}

✅ Consensus reached: Job allocated to GPU cluster with 4/5 votes

⚠️  Phase 2: Fault Injection
🔴 Injecting Byzantine fault into MEMORY_SPECIALIST
🔴 Agent MEMORY_SPECIALIST now provides malicious responses
...

📊 FINAL RESULTS:
Success Rate: 93.3% (28/30 jobs completed)
Byzantine Tolerance: ✅ Maintained correctness under attack
Recovery Time: 15.2s average
LLM API Success: 87.5% (14/16 calls successful)
```

## Educational Value

This demo is designed for:

### Students & Researchers
- Understanding distributed consensus algorithms
- Learning Byzantine fault tolerance principles
- Exploring LLM integration in multi-agent systems
- Studying real-world fault handling strategies

### System Architects
- Evaluating LLM-enhanced decision making
- Testing fault tolerance mechanisms
- Analyzing performance under various failure modes
- Understanding graceful degradation patterns

### HPC Practitioners
- Exploring intelligent resource allocation
- Understanding multi-agent scheduling approaches
- Learning fault-tolerant system design
- Evaluating consensus-driven workflows

## Configuration Options

The demo can be customized by modifying parameters in the script:

```python
# System Configuration
NUM_AGENTS = 5                    # Number of consensus agents
FAULT_INJECTION_RATE = 0.3        # Probability of injecting faults
CONSENSUS_TIMEOUT = 30            # Seconds to wait for votes
LLM_TEMPERATURE = 0.3             # Creativity vs consistency
MAX_TOKENS = 1000                 # LLM response length limit

# Fault Types
BYZANTINE_PROBABILITY = 0.4       # Malicious behavior rate
CRASH_PROBABILITY = 0.3           # Agent failure rate
NETWORK_PARTITION_RATE = 0.2      # Communication failure rate
SLOW_RESPONSE_FACTOR = 3.0        # Response delay multiplier
```

## Troubleshooting

### Common Issues

1. **API Authentication Errors (403 Forbidden)**
   ```bash
   # Verify environment variables
   echo $SAMBASTUDIO_API_KEY
   echo $SAMBASTUDIO_URL
   
   # Source your profile if needed
   source ~/.bashrc  # or ~/.zshrc
   ```

2. **Empty LLM Responses**
   - The demo gracefully handles this with fallback responses
   - Check API rate limits and model availability
   - Consider adjusting temperature settings

3. **JSON Parsing Errors**
   - Demo includes robust parsing with multiple strategies
   - Fallback responses ensure continued operation
   - Check prompt formatting for clarity

### Performance Tuning

- **Reduce Latency**: Lower `MAX_TOKENS` for faster responses
- **Improve Quality**: Increase `LLM_TEMPERATURE` for more creative decisions
- **Adjust Fault Rates**: Modify probability settings to test different scenarios
- **Timeout Settings**: Balance between thoroughness and responsiveness

## Technical Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  LLM Provider   │    │   Consensus      │    │  HPC Resource   │
│  (SambaNova)    │────│   Protocol       │────│   Managers      │
│                 │    │  (Byzantine FT)  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Fault Injection Framework                    │
├─────────────────────────────────────────────────────────────────┤
│  • Byzantine Attacks    • Network Partitions                   │
│  • Crash Failures      • Performance Degradation               │
│  • Message Corruption  • Recovery Simulation                   │
└─────────────────────────────────────────────────────────────────┘
```

## Next Steps

After running this demo, consider exploring:

1. **Advanced Configurations**: Modify agent specializations and resource types
2. **Custom Fault Scenarios**: Add new failure modes relevant to your domain
3. **Performance Analysis**: Collect metrics for research or optimization
4. **Integration**: Adapt the framework for real HPC environments
5. **Scalability Testing**: Increase agent count and job complexity

## Resources

- **Source Code**: `hybrid_llm_demo.py` - Fully commented implementation
- **LangChain Documentation**: [SambaNova Integration Guide](https://python.langchain.com/docs/integrations/llms/sambanova)
- **Byzantine Fault Tolerance**: Academic papers on distributed consensus
- **HPC Scheduling**: Literature on resource allocation algorithms

---

*This demo represents cutting-edge research in LLM-enhanced multi-agent systems for high-performance computing environments. It demonstrates practical applications of artificial intelligence in distributed systems while maintaining educational clarity and research rigor.*

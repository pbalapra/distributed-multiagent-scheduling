# Distributed Agentic AI Consensus Demo

## Overview

This demo showcases a **distributed agentic AI system** where autonomous AI agents manage supercomputer resources through peer-to-peer consensus. The system operates without any central coordinator, demonstrating true decentralized decision-making at exascale computing levels.

## Key Features

### 🌐 Distributed Architecture
- **6 autonomous supercomputer agents** (HPC_RESOURCE_00 through HPC_RESOURCE_05)
- **Peer-to-peer mesh network** with no central authority
- **Byzantine fault tolerance** with 2/3 majority consensus voting
- **Real-time resource allocation** through competitive bidding

### 🧠 AI-Enhanced Decision Making
- **SambaNova LLM integration** (Meta-Llama-3-70B-Instruct) for intelligent bidding
- **Contextual reasoning** about workload requirements and cluster capabilities
- **Fallback heuristics** when LLM is unavailable
- **Reputation-based weighted voting** system

### ⚡ Large Scale Resources
- **1,993 total compute nodes** across all clusters
- **101,984 CPU cores** and **5,136 GPUs**
- **1,108TB total memory** capacity
- **Multi-petabyte storage** systems

## Demo Architecture

### Supercomputer Cluster Specifications

| Cluster | Type | Nodes | CPU Cores | Memory (GB) | GPUs | Interconnect |
|---------|------|-------|-----------|-------------|------|--------------|
| HPC_RESOURCE_00 | hpc | 200 | 8,800 | 102,400 | 1,200 | InfiniBand EDR |
| HPC_RESOURCE_01 | gpu | 320 | 20,480 | 163,840 | 1,280 | Cray Slingshot-11 |
| HPC_RESOURCE_02 | memory | 512 | 24,576 | 524,288 | 0 | Tofu Interconnect D |
| HPC_RESOURCE_03 | hybrid | 328 | 10,496 | 83,968 | 512 | InfiniBand HDR |
| HPC_RESOURCE_04 | ai | 353 | 28,672 | 90,368 | 1,024 | InfiniBand HDR |
| HPC_RESOURCE_05 | storage | 280 | 8,960 | 143,360 | 1,120 | Cray Slingshot-10 |

### Job Scenarios (30-60 nodes each)

1. **Exascale Climate Modeling** (40 nodes) - Weather Research & Forecasting
2. **LLM Training** (60 nodes) - 1T parameter transformer with 3D parallelism  
3. **Cosmological Simulation** (50 nodes) - 100B particle N-body simulation
4. **Quantum Circuit Simulation** (32 nodes) - 45-qubit quantum circuit
5. **Graph Analytics** (42 nodes) - Trillion-edge graph processing
6. **Genomics Analysis** (38 nodes) - 100K whole genome population study
7. **Fusion Plasma Simulation** (36 nodes) - ITER tokamak MHD simulation
8. **Drug Discovery** (40 nodes) - 1M compound molecular docking

## How to Run

```bash
# Basic execution (uses heuristic bidding)
python demos/distributed_agentic_ai_demo.py

# With LLM integration (requires SambaNova API access)
export SAMBASTUDIO_URL="your_sambanova_url"
export SAMBASTUDIO_API_KEY="your_api_key"
python demos/distributed_agentic_ai_demo.py
```

## Demo Output Analysis

### Phase 1: Network Formation
```
🏛️ FORMING SUPERCOMPUTER NETWORK
✅ Added HPC_RESOURCE_00: 200 nodes | 8,800 CPU cores | 102,400GB RAM | 1,200 GPUs
✅ Added HPC_RESOURCE_01: 320 nodes | 20,480 CPU cores | 163,840GB RAM | 1,280 GPUs
...
🌐 Supercomputer network formed: 6 clusters connected
📊 Total Network Capacity: 1,993 compute nodes, 101,984 CPU cores, 1,082TB memory
```

### Phase 2: Job Allocation Process

Each job goes through a **two-phase consensus protocol**:

#### Bidding Phase
```
📋 PHASE 1: CLUSTER BIDDING
📊 HPC_RESOURCE_00: bid=1.000, nodes=200, util=29.6%
📊 HPC_RESOURCE_01: bid=1.000, nodes=320, util=17.1%
...
```

Agents calculate bid scores based on:
- **Resource availability** (can they handle the node count?)
- **Specialization match** (job type vs. cluster type)
- **Current utilization** (workload balancing)
- **Reputation score** (historical performance)

#### Consensus Voting Phase
```
🗳️ PHASE 2: BYZANTINE-TOLERANT CONSENSUS
⚖️ HPC_RESOURCE_00: weighted_score=1.000
...
🏆 PROPOSED WINNER: HPC_RESOURCE_00 (score: 1.000)
🗳️ Requiring 4/6 votes for consensus...
✅ HPC_RESOURCE_00: approve
✅ CONSENSUS REACHED: 6/6 approved
```

Consensus requires **2/3 majority** (4 out of 6 votes) for Byzantine fault tolerance.

### Phase 3: Resource Tracking

After each allocation, the system shows **real-time occupancy**:

```
📈 RESOURCE OCCUPANCY: HPC_RESOURCE_00
🖥️ Nodes: 40/200 (20.0% occupied)
⚡ CPU: 8,800 cores total
💾 Memory: 102,400GB total
🚀 GPU: 1,200 total
🏃 Running Jobs: 1
   • job_001: Weather Research & Forecasting (WRF) (40 nodes, 480min)
```

### Phase 4: Byzantine Fault Injection

The demo demonstrates **fault tolerance** by injecting a Byzantine agent:

```
🚨 SCENARIO 2: BYZANTINE ATTACK ON SUPERCOMPUTERS
🚨 Byzantine fault injected into HPC_RESOURCE_00 (reputation: 0.600)

🗳️ PHASE 2: BYZANTINE-TOLERANT CONSENSUS
🚨 HPC_RESOURCE_00: BYZANTINE PENALTY applied
⚖️ HPC_RESOURCE_00: weighted_score=0.300  # Severely penalized
🛡️ PROTECTED: Byzantine supercomputer was rejected by consensus
```

## LLM Integration Example

When SambaNova API is available, agents use intelligent reasoning:

```
🧠 LLM QUERY FROM HPC_RESOURCE_04
You are HPC_RESOURCE_04, managing a supercomputer cluster in a decentralized 
resource allocation system.

JOB REQUEST (LARGE SCALE):
{
  "job_type": "ai",
  "node_count": 60,
  "application": "Distributed PyTorch Training",
  "model": "1 Trillion parameter transformer"
}

YOUR SUPERCOMPUTER CAPABILITIES:
- Total Nodes: 353
- CPU Cores: 28,672
- Memory: 90,368 GB
- GPUs: 1,024
- Resource Type: ai
- Interconnect: Mellanox InfiniBand HDR (200 Gbps)

💬 SAMBANOVA JSON (0.78s): {
  "bid_score": 0.95, 
  "reasoning": "Perfect match for AI workload with extensive GPU resources and high-bandwidth interconnect optimized for distributed training"
}
```

## Results Summary

The demo processes **8 large multi-node jobs** with the following outcomes:

```
📊 DEMO SUMMARY - 8 SUPERCOMPUTER JOBS PROCESSED
📈 Job Allocation Results:
   ✅ Successful: 8/8 (100.0%)
   ⏱️ Average Consensus Time: 0.00s
📊 Total Resources Allocated: 338 compute nodes

📊 Performance by Scenario:
   🟢 Normal Operations (Jobs 1-6): 6/6 successful
   🔴 Byzantine Attack (Jobs 7-8): 2/2 successful

🛡️ Fault Tolerance Results:
   Byzantine supercomputer: HPC_RESOURCE_00
   Network resilience: ✅ Maintained
```

## Key Innovations

### 1. **True Decentralization**
- No central scheduler or coordinator
- Peer-to-peer decision making
- Autonomous agent behavior

### 2. **AI-Enhanced Resource Management**
- LLM-powered intelligent bidding
- Contextual understanding of workloads
- Adaptive decision making

### 3. **Exascale Demonstration**
- Realistic supercomputer configurations
- Multi-node job allocation (30-60 nodes each)
- Petabyte-scale storage systems

### 4. **Byzantine Fault Tolerance**
- Handles malicious or malfunctioning agents
- 2/3 majority consensus requirement
- Reputation-based trust system

### 5. **Real-time Resource Tracking**
- Live occupancy monitoring
- Running job status
- Dynamic load balancing

## Technical Architecture

### Agent Components
- **Bidding Engine**: Calculates competitive scores for job requests
- **Consensus Protocol**: Implements Byzantine-tolerant voting
- **Resource Manager**: Tracks allocated nodes and running jobs
- **Peer Network**: Maintains mesh connectivity with other agents

### LLM Integration
- **Model**: Meta-Llama-3-70B-Instruct via SambaNova API
- **Reasoning**: Context-aware bid calculation with detailed explanations
- **Fallback**: Heuristic algorithms when LLM unavailable

### Consensus Algorithm
- **Voting Threshold**: 2/3 majority (4 out of 6 agents)
- **Byzantine Penalties**: Malicious agents get severely reduced influence
- **Reputation System**: Historical performance affects voting weight

## Complete Demo Flow

### 1. **Normal Operations (Jobs 1-6)**
```
🎯  Job 1: Exascale Climate Modeling (WRF) - 40 nodes
✅  SUCCESS: Allocated to HPC_RESOURCE_00 in 0.00s
📈 RESOURCE OCCUPANCY: HPC_RESOURCE_00 - 40/200 (20.0% occupied)

🎯  Job 2: LLM Training (1T parameters) - 60 nodes  
✅  SUCCESS: Allocated to HPC_RESOURCE_00 in 0.00s
📈 RESOURCE OCCUPANCY: HPC_RESOURCE_00 - 100/200 (50.0% occupied)
```

Shows intelligent load balancing as agents track resource usage.

### 2. **Byzantine Attack (Jobs 7-8)**
```
🚨 Byzantine fault injected into HPC_RESOURCE_00 (reputation: 0.600)
🎯  Job 7: Fusion Plasma Simulation - 36 nodes (under Byzantine attack)
🛡️ PROTECTED: Byzantine supercomputer was rejected by consensus
✅  SUCCESS: Allocated to HPC_RESOURCE_01 in 0.00s
```

System successfully rejects compromised agent and maintains operations.

### 3. **Final Network Status**
```
📊  NETWORK STATUS
🏛️ Individual Supercomputer Status:
   🚨 BYZANTINE HPC_RESOURCE_00: 200 nodes | Utilization: 59.6% | Reputation: 0.600
   ✅ HEALTHY HPC_RESOURCE_01: 320 nodes | Utilization: 67.1% | Reputation: 1.000
   ...
```

Clear visualization of system health and resource distribution.

## Applications

This distributed agentic AI system demonstrates applications for:

- **Exascale computing resource management**
- **Multi-cloud infrastructure coordination**
- **Autonomous datacenter operations**
- **Decentralized scientific computing**
- **Fault-tolerant distributed systems**

## Future Enhancements

- **Dynamic agent joining/leaving** (network topology changes)
- **Multi-objective optimization** (cost, performance, energy)
- **Hierarchical consensus** for larger networks
- **Machine learning workload prediction**
- **Cross-datacenter federation**
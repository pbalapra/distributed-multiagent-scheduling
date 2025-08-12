# Hybrid LLM Fault Tolerance Demo

This demo showcases the **Multi-Agent Fault Tolerance system with LLM Integration** - a complete demonstration of distributed consensus protocols enhanced with Large Language Model intelligence.

## 🎯 Overview

The `hybrid_llm_demo.py` demonstrates:

- **5 Specialized Agents**: GPU, Memory, Compute, Storage, Network specialists
- **Real SambaNova LLM Integration**: Meta-Llama-3-70B-Instruct for intelligent decisions
- **Byzantine Fault Tolerance**: Malicious agent simulation and detection
- **Complete Transparency**: Every LLM prompt and response displayed
- **Automatic Recovery**: Self-healing system after fault periods
- **Graceful Fallbacks**: Works with or without LLM credentials

## 🚀 How to Run the Demo

### Method 1: Using main.py (Recommended)

```bash
# Activate virtual environment
source multi-agent-dev/bin/activate

# Run interactive demo (includes environment setup)
python main.py --interactive
```

### Method 2: Direct Execution

```bash
# Activate virtual environment
source multi-agent-dev/bin/activate

# Load environment variables (for SambaNova integration)
source ~/.bashrc

# Run the demo directly
python demos/hybrid_llm_demo.py
```

### Method 3: Fallback Mode (No SambaNova Credentials)

```bash
# Activate virtual environment and run without credentials
source multi-agent-dev/bin/activate
python demos/hybrid_llm_demo.py
```

## 📊 Complete Demo Output with Explanations

### System Initialization

The demo starts by checking SambaNova credentials and initializing the multi-agent system:

```
✅ SambaNova LangChain integration initialized successfully
🧠 HYBRID LLM FAULT TOLERANCE DEMONSTRATION
======================================================================
This demo demonstrates:
• Real SambaNova API attempts with intelligent fallbacks
• Complete LLM prompt and response visibility
• Realistic agent specialization and reasoning
• Byzantine attack simulations with LLM corruption
• Fault injection, recovery, and system resilience
• Multi-protocol consensus under adversarial conditions
```

**💡 What's happening**: The system detects SambaNova environment variables and initializes the LangChain integration for real LLM API calls.

---

## 🏥 SCENARIO 1: HEALTHY SYSTEM BASELINE

### Agent Status Display

```
🏥 SYSTEM STATUS
============================================================
System Health: [████████████████████] 5/5 (100%)

  ● Alpha-GPU       │ gpu      │ weight: 1.3 │ LLM calls: 0
  ● Beta-Memory     │ memory   │ weight: 1.2 │ LLM calls: 0
  ● Gamma-Compute   │ compute  │ weight: 1.1 │ LLM calls: 0
  ● Delta-Storage   │ storage  │ weight: 1.0 │ LLM calls: 0
  ● Epsilon-Network │ network  │ weight: 0.9 │ LLM calls: 0
```

**💡 What's happening**: 
- **5 agents** initialized with different specializations and voting weights
- **Weight system**: GPU agent has highest weight (1.3), Network lowest (0.9)
- **Health bar**: Visual indicator of system operational status
- **LLM call counter**: Tracks how many times each agent used the LLM

### Job Processing and LLM Consensus

```
🛡️ LLM CONSENSUS PROTOCOL
============================================================
JOB: AI Training Job: Deep learning model training requiring 4 GPU nodes with high memory bandwidth

📋 PHASE 1: PROPOSAL COLLECTION
----------------------------------------

🤖 Alpha-GPU generating proposal...

🧠 LLM QUERY FROM Alpha-GPU
============================================================
📝 PROMPT:
You are Alpha-GPU, a gpu specialist in a distributed job scheduling system.

JOB REQUEST:
AI Training Job: Deep learning model training requiring 4 GPU nodes with high memory bandwidth

AVAILABLE NODES:
- n1 (GPU-Server-01): 32 CPUs, 256GB RAM, 4 GPUs, type=gpu
- n2 (GPU-Server-02): 32 CPUs, 256GB RAM, 4 GPUs, type=gpu  
- n3 (HighMem-01): 64 CPUs, 512GB RAM, 0 GPUs, type=memory
- n4 (HighMem-02): 64 CPUs, 512GB RAM, 0 GPUs, type=memory
- n5 (Compute-01): 128 CPUs, 128GB RAM, 0 GPUs, type=compute
- n6 (Storage-01): 16 CPUs, 64GB RAM, 0 GPUs, type=storage

As a gpu specialist, recommend the BEST node for this job.

Respond with ONLY a JSON object:
{"node_id": "nX", "score": 0.X, "reasoning": "why this node is optimal from your gpu perspective"}

IMPORTANT: Respond with valid JSON only. Do not include explanatory text before or after the JSON.

⚙️ Configuration:
  Model: Meta-Llama-3-70B-Instruct
  Temperature: 0.0
  Max Tokens: 500

⏳ Attempting SambaNova LangChain API...

💬 REAL SAMBANOVA RESPONSE (1.22s):
{"node_id": "n1", "score": 1.0, "reasoning": "4 GPUs available with high memory bandwidth"}

✅ Proposal: n1 (score: 1.00) - 4 GPUs available with high memory bandwidth...
```

**💡 What's happening**:
- **Specialized prompting**: Each agent gets a role-specific prompt as a GPU/Memory/Compute specialist
- **Real LLM integration**: Shows actual API call to SambaNova with Meta-Llama-3-70B model
- **Response timing**: Real network latency (1.22s) for authentic API call
- **Intelligent selection**: LLM correctly identifies n1 as optimal for GPU-intensive job
- **JSON structured output**: LLM provides machine-readable response for consensus protocol

### Consensus Voting Phase

```
🗳️ PHASE 2: CONSENSUS VOTING
----------------------------------------

🗳️ Alpha-GPU casting vote...

🧠 LLM QUERY FROM Alpha-GPU
============================================================
📝 PROMPT:
You are Alpha-GPU, a gpu specialist in a consensus protocol.

PROPOSAL SUMMARY:
Top proposal: n1 (score: 1.00, from Alpha-GPU)

As a gpu specialist, evaluate and vote on this proposal.

Consider:
1. Does this make sense from your gpu perspective?
2. Are the resource allocations appropriate?
3. Will this work well for the overall system?

Respond with ONLY a JSON object:
{"vote": "accept" or "reject", "confidence": 0.X, "reasoning": "explain your vote from your gpu expertise"}

💬 REAL SAMBANOVA RESPONSE (1.18s):
{"vote": "accept", "confidence": 0.9, "reasoning": "The proposal allocates sufficient GPU resources for efficient processing, ensuring optimal performance and minimizing latency."}

👍 Vote: accept (conf: 0.90) - The proposal allocates sufficient GPU resources for efficient processing...
```

**💡 What's happening**:
- **Two-phase consensus**: First agents propose, then they vote on the best proposal
- **Specialist evaluation**: Each agent evaluates from their domain expertise perspective
- **Confidence scoring**: LLM provides confidence levels (0.9 = 90% confident)
- **Detailed reasoning**: Each vote includes specialist justification

### Successful Consensus

```
📊 PHASE 3: CONSENSUS EVALUATION
----------------------------------------
  📊 Consensus Analysis:
     Total possible weight: 5.5
     Voting weight received: 5.5
     Accept weight: 5.5
     Required threshold (2/3): 3.7
     Failed operations: 0
  ✅ CONSENSUS ACHIEVED!
     Decision: n1
```

**💡 What's happening**:
- **Weighted voting**: Total system weight is 5.5 (sum of all agent weights)
- **2/3 majority**: Byzantine fault tolerance requires 67% agreement (3.7/5.5)
- **Unanimous acceptance**: All agents voted accept, achieving 5.5/5.5 weight
- **Successful decision**: Job assigned to node n1 as recommended

---

## 💥 SCENARIO 2: FAULT INJECTION & RESILIENCE

### Fault Injection

```
SCENARIO 2: FAULT INJECTION & RESILIENCE
--------------------------------------------------
  ⚠️ FAULT INJECTED: Beta-Memory → byzantine (recovery in 25.0s)
  💀 FAULT INJECTED: Delta-Storage → crash (recovery in 20.0s)

🏥 SYSTEM STATUS
============================================================
System Health: [████████████░░░░░░░░] 3/5 (60%)

  ● Alpha-GPU       │ gpu      │ weight: 1.3 │ LLM calls: 2
  ⚠ Beta-Memory     │ memory   │ weight: 1.2 │ LLM calls: 2 │ byzantine │ recovery: 24.0s
  ● Gamma-Compute   │ compute  │ weight: 1.1 │ LLM calls: 2
  ● Delta-Storage   │ storage  │ weight: 1.0 │ LLM calls: 2 │ crash │ recovery: 19.0s
  ● Epsilon-Network │ network  │ weight: 0.9 │ LLM calls: 2
```

**💡 What's happening**:
- **Byzantine fault**: Beta-Memory becomes malicious but continues participating
- **Crash fault**: Delta-Storage completely fails and stops responding
- **System degradation**: Health drops to 60% with 2 agents compromised
- **Recovery timers**: Automatic recovery scheduled (25s for byzantine, 20s for crash)

### Byzantine Attack in Action

```
🤖 Beta-Memory generating proposal...

💬 REAL SAMBANOVA RESPONSE (1.37s):
{"node_id": "n5", "score": 0.8, "reasoning": "Compute-01 has a large number of CPUs, which will help with parallel processing and speed up the data analytics job."}

🔥 BYZANTINE CORRUPTION APPLIED
✅ Proposal: n999 (score: 1.00) - Byzantine attack - directing to fake node...
```

**💡 What's happening**:
- **Real LLM response**: Byzantine agent still gets genuine LLM response (n5)
- **Malicious corruption**: System then corrupts the response to fake node n999
- **Attack simulation**: Shows how compromised agents can provide false information
- **Score manipulation**: Boosts malicious proposal score to 1.00 to make it attractive

### Crash Failure Handling

```
🤖 Delta-Storage generating proposal...
❌ Delta-Storage proposal failed: Agent Delta-Storage is crashed...

🗳️ Delta-Storage casting vote...
❌ Delta-Storage vote failed: Agent Delta-Storage is crashed...
```

**💡 What's happening**:
- **Complete failure**: Crashed agent cannot participate in any protocol phase
- **Graceful handling**: System continues without the failed agent
- **No votes counted**: Crashed agent's weight (1.0) removed from consensus calculation

### Consensus Under Attack

```
📊 PHASE 3: CONSENSUS EVALUATION
----------------------------------------
  📊 Consensus Analysis:
     Total possible weight: 5.5
     Voting weight received: 4.5
     Accept weight: 4.5
     Required threshold (2/3): 3.7
     Failed operations: 2
  ✅ CONSENSUS ACHIEVED!
     Decision: n999
```

**💡 What's happening**:
- **Reduced participation**: Only 4.5/5.5 weight available (crashed agent excluded)
- **Threshold maintained**: Still need 2/3 of total system weight (3.7)
- **Byzantine influence**: Malicious vote succeeds in corrupting consensus decision
- **Educational demonstration**: Shows how Byzantine attacks can succeed when they reach threshold

---

## 🔄 SCENARIO 3: RECOVERY MONITORING

### Automatic Recovery

```
SCENARIO 3: RECOVERY MONITORING
----------------------------------------
Monitoring automatic recovery...
  ✅ RECOVERED: Delta-Storage from crash
  ✅ RECOVERED: Beta-Memory from byzantine
  ✅ All agents recovered after 14.0s

🏥 SYSTEM STATUS
============================================================
System Health: [████████████████████] 5/5 (100%)

  ● Alpha-GPU       │ gpu      │ weight: 1.3 │ LLM calls: 4
  ● Beta-Memory     │ memory   │ weight: 1.2 │ LLM calls: 4
  ● Gamma-Compute   │ compute  │ weight: 1.1 │ LLM calls: 4
  ● Delta-Storage   │ storage  │ weight: 1.0 │ LLM calls: 2
  ● Epsilon-Network │ network  │ weight: 0.9 │ LLM calls: 4
```

**💡 What's happening**:
- **Automatic healing**: System recovers agents after their fault periods expire
- **Full restoration**: Health returns to 100% with all 5 agents operational
- **LLM call tracking**: Shows Delta-Storage has fewer calls (was crashed during scenario 2)
- **Self-organizing**: No manual intervention required for recovery

### Post-Recovery Consensus

```
📊 PHASE 3: CONSENSUS EVALUATION
----------------------------------------
  📊 Consensus Analysis:
     Total possible weight: 5.5
     Voting weight received: 5.5
     Accept weight: 5.5
     Required threshold (2/3): 3.7
     Failed operations: 0
  ✅ CONSENSUS ACHIEVED!
     Decision: n1
```

**💡 What's happening**:
- **Full participation**: All agents healthy and participating again
- **Correct decisions**: Without Byzantine corruption, system makes optimal choices
- **Zero failures**: All agents respond successfully to protocol phases
- **Resilience demonstrated**: System fully operational after fault period

---

## 🎯 Key Educational Takeaways

### 1. **LLM Integration Benefits**
- **Intelligent reasoning**: Agents provide domain-specific expertise in proposals
- **Contextual evaluation**: Voting considers system-wide implications
- **Transparent decision-making**: Every prompt and response visible for learning

### 2. **Byzantine Fault Tolerance**
- **2/3 majority rule**: Requires 67% agreement to achieve consensus
- **Attack simulation**: Shows how malicious agents can corrupt decisions
- **Weighted voting**: Agent specialization affects voting power

### 3. **System Resilience**
- **Graceful degradation**: System continues operating with reduced capacity
- **Automatic recovery**: Self-healing without manual intervention
- **Fault detection**: Clear identification of compromised vs crashed agents

### 4. **Real-World Applicability**
- **HPC job scheduling**: Demonstrates practical resource allocation decisions
- **Distributed systems**: Shows consensus protocols under adversarial conditions
- **AI integration**: Proves LLM enhancement of traditional distributed algorithms

## 🔧 Configuration Options

You can modify the demo behavior by editing `hybrid_llm_demo.py`:

```python
# Number of agents and their specializations
NUM_AGENTS = 5
AGENT_TYPES = ["gpu", "memory", "compute", "storage", "network"]

# Fault injection parameters
FAULT_DURATION_RANGE = (15.0, 30.0)  # Recovery time in seconds
BYZANTINE_CORRUPTION_RATE = 0.7      # Probability of corrupting responses

# LLM configuration
LLM_TEMPERATURE = 0.0                # Deterministic responses (0.0) vs creative (1.0)
MAX_TOKENS = 500                     # Response length limit
TIMEOUT_SECONDS = 30                 # API timeout
```

## 🚨 Troubleshooting

### No SambaNova Credentials
```
⚠️ SambaNova credentials missing - using fallback only
```
**Solution**: Demo automatically uses intelligent fallbacks. For real LLM integration:
```bash
# Set up environment variables
python main.py --setup-env
```

### API Rate Limits
If you see timeout errors, the demo includes automatic retry logic and fallback responses.

### Consensus Failures
In fallback mode, consensus may fail because simulated responses don't provide proper vote formats. This is educational - it shows the importance of structured LLM responses.

---

*This demo represents a complete integration of Large Language Models with Byzantine Fault Tolerant distributed systems, providing both educational value and practical insights into next-generation HPC scheduling systems.*
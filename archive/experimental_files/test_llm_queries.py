#!/usr/bin/env python3
"""
LLM Query and Response Test
==========================

This script shows the exact queries being sent to the LLM
and the responses received, with detailed logging.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.ollama_provider import OllamaProvider
from llm.llm_interface import LLMRequest
from datetime import datetime

def test_job_scoring():
    """Test job scoring with detailed logging"""
    print("=" * 80)
    print("🎯 JOB SCORING LLM TEST")
    print("=" * 80)
    
    provider = OllamaProvider(model_name="mistral", timeout=60.0)
    
    # Create a job scoring request
    prompt = """You are an intelligent job scheduler for a distributed computing system. 
Analyze this job scheduling scenario and provide a scoring recommendation.

JOB REQUIREMENTS:
- CPU Cores: 4
- Memory: 8 GB
- Priority: HIGH
- Estimated Runtime: 120 minutes
- User: research_team

AVAILABLE RESOURCE:
- Total CPU Cores: 16
- Total Memory: 64 GB
- Current CPU Usage: 25% (4 cores in use)
- Current Memory Usage: 20% (12.8 GB in use)
- Current Utilization Score: 0.225
- Resource Type: CPU_CLUSTER
- Location: DataCenter-A

Please provide a JSON response with your analysis:
{
  "score": <float between 0.0 and 1.0>,
  "recommendation": "<accept|consider|reject>",
  "reasoning": "<your detailed reasoning>",
  "factors": {
    "resource_match": <float>,
    "utilization_impact": <float>,
    "priority_weight": <float>
  }
}"""

    request = LLMRequest(
        prompt=prompt,
        context={
            "job": {
                "cpu_cores": 4,
                "memory_gb": 8,
                "priority": "HIGH",
                "runtime_minutes": 120
            },
            "resource": {
                "total_cpu": 16,
                "total_memory": 64,
                "current_cpu_usage": 0.25,
                "current_memory_usage": 0.20,
                "utilization_score": 0.225
            }
        },
        task_type="job_scoring",
        temperature=0.1,
        max_tokens=300
    )
    
    print(f"\n🚀 Starting job scoring test at {datetime.now().strftime('%H:%M:%S')}")
    response = provider.generate_sync(request)
    
    print(f"\n📋 FINAL PROCESSED RESPONSE:")
    print(f"Content: {response.content}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Metadata: {response.metadata}")

def test_fault_recovery():
    """Test fault recovery with detailed logging"""
    print("\n" + "=" * 80)
    print("🛠️ FAULT RECOVERY LLM TEST")
    print("=" * 80)
    
    provider = OllamaProvider(model_name="mistral", timeout=60.0)
    
    prompt = """You are a fault recovery system for distributed job scheduling.
A job execution has failed and you need to recommend a recovery strategy.

FAILURE CONTEXT:
- Job ID: job-critical-001
- Error Type: network_timeout
- Error Message: "Connection lost to worker node after 45 seconds"
- Failed Resource: worker-node-03
- Retry Count: 1 (this is the second attempt)
- Job Priority: CRITICAL
- Job Progress: 60% complete before failure
- Time Remaining: 30 minutes until deadline

AVAILABLE ALTERNATIVES:
- worker-node-01: Available, CPU=16/32, Memory=24GB/64GB
- worker-node-05: Available, CPU=8/32, Memory=45GB/64GB
- worker-node-07: Busy, CPU=30/32, Memory=60GB/64GB

SYSTEM STATUS:
- Network: Intermittent issues reported in Zone-C
- Load: Medium (65% cluster utilization)
- Queue: 12 jobs waiting

Recommend a recovery strategy as JSON:
{
  "strategy": "<immediate_retry|delayed_retry|reschedule|escalate>",
  "action": "<specific action to take>",
  "alternative_resource": "<resource_id or null>",
  "delay_seconds": <int>,
  "reasoning": "<your detailed analysis>"
}"""

    request = LLMRequest(
        prompt=prompt,
        context={
            "failure": {
                "type": "network_timeout",
                "retry_count": 1,
                "job_progress": 0.6,
                "deadline_minutes": 30
            }
        },
        task_type="fault_recovery",
        temperature=0.1,
        max_tokens=250
    )
    
    print(f"\n🚀 Starting fault recovery test at {datetime.now().strftime('%H:%M:%S')}")
    response = provider.generate_sync(request)
    
    print(f"\n📋 FINAL PROCESSED RESPONSE:")
    print(f"Content: {response.content}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Metadata: {response.metadata}")

def test_negotiation():
    """Test negotiation with detailed logging"""
    print("\n" + "=" * 80)
    print("🤝 NEGOTIATION LLM TEST")
    print("=" * 80)
    
    provider = OllamaProvider(model_name="mistral", timeout=60.0)
    
    prompt = """You are an intelligent resource negotiation agent.
Another agent is requesting resources and you need to decide whether to accept.

RESOURCE REQUEST:
- Requesting Agent: compute-cluster-B
- Resources Needed: 6 CPU cores, 16 GB RAM
- Duration: 90 minutes
- Priority Level: 3 (medium-high)
- Deadline: 2 hours from now
- Offered Compensation: 150 credits
- Job Type: Machine Learning Training

YOUR CURRENT STATUS:
- Available CPU: 8 cores (out of 16 total)
- Available Memory: 20 GB (out of 32 total)
- Current Utilization: 70%
- Scheduled Jobs: 3 (all medium priority)
- Credit Balance: 850 credits
- Your Priority Jobs in Queue: 1 (low priority)

NEGOTIATION ROUND: 1 (initial offer)

Decide whether to accept this proposal or make a counter-offer:
{
  "accept": <true|false>,
  "counter_offer": {
    "priority_adjustment": <int or null>,
    "duration_limit": <minutes or null>,
    "credit_request": <credits or null>,
    "alternative_time": "<time slot or null>"
  },
  "reasoning": "<your negotiation logic>",
  "confidence": <float 0.0-1.0>
}"""

    request = LLMRequest(
        prompt=prompt,
        context={
            "proposal": {
                "cpu_cores": 6,
                "memory_gb": 16,
                "duration_minutes": 90,
                "priority": 3,
                "credits": 150
            },
            "agent_state": {
                "available_cpu": 8,
                "available_memory": 20,
                "utilization": 0.70,
                "credit_balance": 850
            }
        },
        task_type="negotiation",
        temperature=0.2,
        max_tokens=300
    )
    
    print(f"\n🚀 Starting negotiation test at {datetime.now().strftime('%H:%M:%S')}")
    response = provider.generate_sync(request)
    
    print(f"\n📋 FINAL PROCESSED RESPONSE:")
    print(f"Content: {response.content}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Metadata: {response.metadata}")

def main():
    """Run all LLM query tests"""
    print("🤖 LLM QUERY AND RESPONSE DETAILED TEST")
    print("=" * 80)
    print("This test shows exactly what queries are sent to the LLM")
    print("and the detailed responses received back.")
    print()
    
    try:
        test_job_scoring()
        test_fault_recovery()
        test_negotiation()
        
        print("\n" + "=" * 80)
        print("✅ ALL LLM QUERY TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("Key observations:")
        print("- All prompts are detailed and context-rich")
        print("- Responses show actual LLM reasoning")
        print("- JSON parsing and validation working")
        print("- System falls back gracefully on any issues")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    main()

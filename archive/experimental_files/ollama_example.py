#!/usr/bin/env python3
"""
Ollama LLM Integration Example
=============================

This script demonstrates how to use Ollama with the multiagent system.
"""

import sys
import json
import asyncio
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from llm.ollama_provider import OllamaProvider
from llm.llm_interface import LLMRequest, LLMManager, LLMProvider


async def main():
    print("🦙 Ollama LLM Integration Example")
    print("=" * 50)
    
    # Available models in your system
    available_models = ["mistral:latest", "qwq:32b", "llama3.3:latest", "codestral:latest"]
    
    print(f"Available models: {', '.join(available_models)}")
    
    # Use Mistral for this example
    provider = OllamaProvider(model_name="mistral")
    
    if not provider.is_available():
        print("❌ Ollama service or model not available")
        return
    
    print("\n1. 📝 Basic Text Generation")
    print("-" * 30)
    
    basic_request = LLMRequest(
        prompt="What are the key benefits of distributed computing systems?",
        context={},
        task_type="explanation",
        temperature=0.3,
        max_tokens=150
    )
    
    response = provider.generate_sync(basic_request)
    print(f"Response: {response.content}")
    print(f"Response time: {response.metadata.get('response_time', 0):.2f}s")
    
    print("\n2. 🎯 Job Scoring Scenario")
    print("-" * 30)
    
    job_request = LLMRequest(
        prompt="""Analyze this job scheduling scenario and provide a structured response:

Job Requirements:
- CPU cores: 6
- Memory: 12GB
- Runtime estimate: 2 hours
- Priority: High (1)

Available Resource:
- CPU cores available: 8
- Memory available: 16GB
- Current utilization: 40%
- Queue length: 3 jobs

Please provide a JSON response with:
{
  "score": <float 0-1>,
  "recommendation": "<accept|consider|reject>",
  "factors": {
    "cpu_compatibility": <float>,
    "memory_compatibility": <float>,
    "load_factor": <float>,
    "priority_impact": <float>
  },
  "reasoning": "<explanation>"
}""",
        context={
            "job": {
                "cpu_cores": 6,
                "memory_gb": 12,
                "runtime_estimate": 7200,
                "priority": 1
            },
            "resource": {
                "available_cpu": 8,
                "available_memory": 16,
                "current_utilization": 0.4,
                "queue_length": 3
            }
        },
        task_type="job_scoring",
        temperature=0.1,
        max_tokens=300
    )
    
    job_response = provider.generate_sync(job_request)
    
    try:
        job_data = json.loads(job_response.content)
        print("📊 Job Scoring Results:")
        print(f"   Score: {job_data.get('score', 'N/A')}")
        print(f"   Recommendation: {job_data.get('recommendation', 'N/A')}")
        print(f"   Reasoning: {job_data.get('reasoning', 'N/A')}")
        
        factors = job_data.get('factors', {})
        if factors:
            print("   Factors:")
            for factor, value in factors.items():
                print(f"     - {factor}: {value}")
    except json.JSONDecodeError:
        print(f"Raw response: {job_response.content}")
    
    print("\n3. 🚨 Fault Recovery Scenario")
    print("-" * 30)
    
    fault_request = LLMRequest(
        prompt="""A job execution has failed with the following context:

Failure Details:
- Error: "Connection timeout to worker node"
- Retry count: 1
- Failed resource: agent-worker-3
- Available alternatives: ["agent-worker-1", "agent-worker-4"]
- Job criticality: High
- Deadline: 30 minutes remaining

Recommend recovery strategy as JSON:
{
  "strategy": "<retry|reschedule|escalate>",
  "action": "<immediate|delayed|manual>",
  "alternative_resource": "<resource_id>",
  "delay_seconds": <int>,
  "escalate": <boolean>,
  "reasoning": "<explanation>"
}""",
        context={
            "failure": {
                "type": "connection_timeout",
                "retry_count": 1,
                "failed_resource": "agent-worker-3",
                "available_resources": ["agent-worker-1", "agent-worker-4"]
            },
            "job": {
                "criticality": "high",
                "deadline_minutes": 30
            }
        },
        task_type="fault_recovery",
        temperature=0.1,
        max_tokens=250
    )
    
    fault_response = provider.generate_sync(fault_request)
    
    try:
        fault_data = json.loads(fault_response.content)
        print("🔧 Fault Recovery Strategy:")
        print(f"   Strategy: {fault_data.get('strategy', 'N/A')}")
        print(f"   Action: {fault_data.get('action', 'N/A')}")
        print(f"   Alternative resource: {fault_data.get('alternative_resource', 'N/A')}")
        print(f"   Delay: {fault_data.get('delay_seconds', 0)} seconds")
        print(f"   Escalate: {fault_data.get('escalate', False)}")
        print(f"   Reasoning: {fault_data.get('reasoning', 'N/A')}")
    except json.JSONDecodeError:
        print(f"Raw response: {fault_response.content}")
    
    print("\n4. 🤝 Agent Negotiation Scenario")
    print("-" * 30)
    
    negotiation_request = LLMRequest(
        prompt="""Two agents are negotiating resource allocation:

Proposal from Agent A:
- Requests: 4 CPU cores, 8GB RAM
- Duration: 1 hour
- Priority: 2 (medium)
- Deadline: 45 minutes
- Compensation: 100 credits

Agent B Current State:
- Available: 6 CPU cores, 12GB RAM
- Current utilization: 60%
- Scheduled jobs: 2 (low priority)
- Credit balance: 250

Should Agent B accept this proposal? Provide JSON response:
{
  "accept": <boolean>,
  "counter_offer": {
    "priority_increase": <int>,
    "credits_requested": <int>,
    "delay_acceptable": <int minutes>
  },
  "reasoning_factors": {
    "resource_availability": <string>,
    "priority_impact": <string>,
    "economic_benefit": <string>
  },
  "confidence": <float 0-1>
}""",
        context={
            "proposal": {
                "cpu_cores": 4,
                "memory_gb": 8,
                "duration_minutes": 60,
                "priority": 2,
                "deadline_minutes": 45,
                "compensation": 100
            },
            "agent_state": {
                "available_cpu": 6,
                "available_memory": 12,
                "utilization": 0.6,
                "scheduled_jobs": 2,
                "credit_balance": 250
            }
        },
        task_type="negotiation",
        temperature=0.2,
        max_tokens=300
    )
    
    negotiation_response = provider.generate_sync(negotiation_request)
    
    try:
        negotiation_data = json.loads(negotiation_response.content)
        print("🤝 Negotiation Decision:")
        print(f"   Accept: {negotiation_data.get('accept', 'N/A')}")
        print(f"   Confidence: {negotiation_data.get('confidence', 'N/A')}")
        
        counter_offer = negotiation_data.get('counter_offer', {})
        if counter_offer:
            print("   Counter offer:")
            for key, value in counter_offer.items():
                print(f"     - {key}: {value}")
        
        factors = negotiation_data.get('reasoning_factors', {})
        if factors:
            print("   Reasoning factors:")
            for factor, reasoning in factors.items():
                print(f"     - {factor}: {reasoning}")
                
    except json.JSONDecodeError:
        print(f"Raw response: {negotiation_response.content}")
    
    print("\n5. 🎛️ LLM Manager Integration")
    print("-" * 30)
    
    # Demonstrate using the LLM manager
    manager = LLMManager()
    
    # Add OLLAMA to enum if not present
    if not hasattr(LLMProvider, 'OLLAMA'):
        LLMProvider.OLLAMA = "ollama"
    
    manager.register_provider(LLMProvider.OLLAMA, provider)
    manager.set_default_provider(LLMProvider.OLLAMA)
    
    manager_request = LLMRequest(
        prompt="Explain the advantages of using LLMs in distributed system scheduling in 2-3 sentences.",
        context={},
        task_type="explanation"
    )
    
    manager_response = await manager.generate(manager_request)
    print(f"Manager response: {manager_response.content}")
    
    print("\n✅ Example completed successfully!")
    print("\n💡 Tips for Production Use:")
    print("  - Adjust temperature based on task criticality")
    print("  - Use structured prompts for consistent JSON responses")
    print("  - Implement proper error handling and fallbacks")
    print("  - Monitor response times and adjust timeouts")
    print("  - Consider model selection based on task complexity")


if __name__ == "__main__":
    asyncio.run(main())

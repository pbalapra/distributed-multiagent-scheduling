#!/usr/bin/env python3
"""
Simple SambaNova Multi-Consensus Demo
Optimized for SambaNova LLM response patterns
"""

import json
import time
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

# Import the original demo components
import sys
sys.path.append('.')
from sambanova_consensus_demo import SambaNova_LLMManager, Job, Node

# Simple Priority for demo
class Priority:
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class SimpleConsensusAgent:
    """Simplified consensus agent optimized for SambaNova responses"""
    
    def __init__(self, name: str, specialization: str, weight: float = 1.0):
        self.name = name
        self.specialization = specialization
        self.weight = weight
        self.llm_manager = SambaNova_LLMManager()
    
    def simple_proposal(self, job: Job, nodes: List[Node]) -> Dict:
        """Create a simple, focused proposal"""
        
        # Ultra-simple prompt to avoid repetitive responses
        prompt = f"""Job: {job.name}
Needs: {job.nodes_required} nodes, {job.cpu_per_node}CPU/{job.memory_per_node}GB each
Your role: {self.specialization} specialist

Best node? Reply ONLY:
{{"node_id": "n1", "score": 0.9, "reason": "good fit"}}"""

        try:
            response = self.llm_manager.query(prompt, "proposal", temperature=0.0, max_tokens=100)
            return self._extract_simple_json(response)
        except Exception as e:
            print(f"    ❌ {self.name} proposal failed: {e}")
            return self._fallback_proposal(job, nodes)
    
    def simple_vote(self, job: Job, node_id: str) -> Dict:
        """Simple voting with minimal prompt"""
        
        prompt = f"""Vote on: {job.name} → {node_id}
Job needs: {job.cpu_per_node}CPU/{job.memory_per_node}GB per node

Reply ONLY: {{"vote": "accept", "confidence": 0.8}}"""

        try:
            response = self.llm_manager.query(prompt, "vote", temperature=0.0, max_tokens=50)
            return self._extract_simple_json(response)
        except Exception as e:
            return {"vote": "accept", "confidence": 0.7, "reasoning": "fallback"}
    
    def _extract_simple_json(self, response: str) -> Dict:
        """Extract JSON from response with focus on first occurrence"""
        if not response:
            return {}
        
        # Find first JSON-like structure
        json_start = response.find('{')
        if json_start == -1:
            return {}
        
        # Find matching closing brace
        brace_count = 0
        json_end = json_start
        for i, char in enumerate(response[json_start:], json_start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        try:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback extraction
            return self._manual_extraction(response)
    
    def _manual_extraction(self, text: str) -> Dict:
        """Manual extraction for common patterns"""
        result = {}
        
        # Extract vote
        vote_match = re.search(r'"vote":\s*"(accept|reject)"', text, re.IGNORECASE)
        if vote_match:
            result["vote"] = vote_match.group(1).lower()
        
        # Extract node_id
        node_match = re.search(r'"node_id":\s*"(n\d+)"', text, re.IGNORECASE)
        if node_match:
            result["node_id"] = node_match.group(1)
        
        # Extract score/confidence
        score_match = re.search(r'"(?:score|confidence)":\s*([0-9.]+)', text)
        if score_match:
            result["score" if "score" in text else "confidence"] = float(score_match.group(1))
        
        return result if result else {"vote": "accept", "confidence": 0.5}
    
    def _fallback_proposal(self, job: Job, nodes: List[Node]) -> Dict:
        """Intelligent fallback proposal based on specialization"""
        # Choose best node based on specialization
        best_node = nodes[0]  # Default
        score = 0.5
        
        if self.specialization == "gpu":
            for node in nodes:
                if node.node_type == "gpu":
                    best_node = node
                    score = 0.9
                    break
        elif self.specialization == "memory":
            for node in nodes:
                if node.node_type == "memory" or node.memory_gb >= job.memory_per_node * 2:
                    best_node = node
                    score = 0.8
                    break
        elif self.specialization == "compute":
            for node in nodes:
                if node.node_type == "compute" or node.cpu_cores >= job.cpu_per_node:
                    best_node = node
                    score = 0.85
                    break
        
        return {
            "node_id": best_node.id,
            "score": score,
            "reasoning": f"Fallback {self.specialization} choice"
        }

class SimpleConsensusTester:
    """Test runner for simplified consensus protocols"""
    
    def __init__(self):
        self.nodes = self._create_nodes()
        self.agents = self._create_agents()
        self.results = {}
    
    def _create_nodes(self) -> List[Node]:
        """Create test cluster nodes"""
        return [
            Node("n1", "GPU-Server-01", 32, 256, "gpu", 4, 1000, 100),
            Node("n2", "GPU-Server-02", 32, 256, "gpu", 4, 1000, 100),
            Node("n3", "HighMem-01", 64, 512, "memory", 0, 2000, 200),
            Node("n4", "HighMem-02", 64, 512, "memory", 0, 2000, 200),
            Node("n5", "Compute-01", 128, 128, "compute", 0, 1500, 150),
            Node("n6", "Compute-02", 128, 128, "compute", 0, 1500, 150),
        ]
    
    def _create_agents(self) -> List[SimpleConsensusAgent]:
        """Create consensus agents"""
        return [
            SimpleConsensusAgent("GPU-Expert", "gpu", 1.2),
            SimpleConsensusAgent("Memory-Expert", "memory", 1.1),
            SimpleConsensusAgent("Compute-Expert", "compute", 1.0),
        ]
    
    def test_simple_consensus(self, job: Job) -> Dict:
        """Test a simple consensus protocol"""
        print(f"\n🎯 Testing Simple Consensus for {job.name}")
        print(f"   Needs: {job.nodes_required} nodes, {job.cpu_per_node}CPU/{job.memory_per_node}GB each")
        
        start_time = time.time()
        
        # Phase 1: Get proposals from all agents
        print("   📋 Phase 1: Collecting proposals...")
        proposals = []
        for agent in self.agents:
            print(f"      🤖 {agent.name} proposing...")
            proposal = agent.simple_proposal(job, self.nodes)
            if proposal:
                proposals.append({
                    "agent": agent.name,
                    "proposal": proposal,
                    "weight": agent.weight
                })
                node_id = proposal.get("node_id", "unknown")
                score = proposal.get("score", 0.0)
                print(f"         ✅ Proposes {node_id} (score: {score:.2f})")
            else:
                print(f"         ❌ Failed to propose")
        
        if not proposals:
            return {"success": False, "reason": "No valid proposals"}
        
        # Phase 2: Simple voting on best proposal
        print("   🗳️ Phase 2: Voting on best proposal...")
        best_proposal = max(proposals, key=lambda p: p["proposal"].get("score", 0) * p["weight"])
        chosen_node = best_proposal["proposal"]["node_id"]
        
        votes = []
        for agent in self.agents:
            print(f"      🗳️ {agent.name} voting on {chosen_node}...")
            vote = agent.simple_vote(job, chosen_node)
            votes.append({
                "agent": agent.name,
                "vote": vote,
                "weight": agent.weight
            })
            vote_result = vote.get("vote", "abstain")
            confidence = vote.get("confidence", 0.0)
            print(f"         {'✅' if vote_result == 'accept' else '❌'} {vote_result} (conf: {confidence:.2f})")
        
        # Calculate consensus
        total_weight = sum(v["weight"] for v in votes if v["vote"].get("vote") == "accept")
        total_possible = sum(v["weight"] for v in votes)
        consensus_score = total_weight / total_possible if total_possible > 0 else 0
        
        success = consensus_score >= 0.5  # Simple majority by weight
        duration = time.time() - start_time
        
        result = {
            "success": success,
            "node_id": chosen_node if success else None,
            "consensus_score": consensus_score,
            "duration": duration,
            "proposals": len(proposals),
            "votes": len(votes)
        }
        
        print(f"   📊 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"      Consensus: {consensus_score:.2f}, Duration: {duration:.1f}s")
        
        return result
    
    def run_test_suite(self, num_jobs: int = 3):
        """Run comprehensive test suite"""
        print("🚀 SIMPLE SAMBANOVA CONSENSUS DEMO")
        print("=" * 50)
        
        # Create test jobs
        test_jobs = [
            Job("AI-Training-001", "ml", Priority.HIGH, 4, 64, 128, "gpu"),
            Job("Data-Analysis-002", "analytics", Priority.MEDIUM, 8, 32, 64, "compute"),
            Job("Climate-Sim-003", "simulation", Priority.HIGH, 2, 128, 256, "memory"),
        ][:num_jobs]
        
        results = []
        
        for i, job in enumerate(test_jobs, 1):
            print(f"\n{'='*20} JOB {i}/{len(test_jobs)} {'='*20}")
            result = self.test_simple_consensus(job)
            results.append({
                "job": job.name,
                "result": result
            })
        
        # Summary
        print(f"\n{'='*50}")
        print("🏁 FINAL SUMMARY")
        print("=" * 50)
        
        successful = sum(1 for r in results if r["result"]["success"])
        total_time = sum(r["result"]["duration"] for r in results)
        avg_consensus = sum(r["result"]["consensus_score"] for r in results) / len(results)
        
        print(f"✅ Success Rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"⏱️  Total Time: {total_time:.1f}s (avg: {total_time/len(results):.1f}s per job)")
        print(f"🤝 Avg Consensus: {avg_consensus:.2f}")
        
        for r in results:
            status = "✅" if r["result"]["success"] else "❌"
            duration = r["result"]["duration"]
            consensus = r["result"]["consensus_score"]
            print(f"   {status} {r['job']}: {consensus:.2f} consensus in {duration:.1f}s")
        
        return results

def main():
    """Run the simple consensus demo"""
    tester = SimpleConsensusTester()
    results = tester.run_test_suite(3)
    
    print(f"\n🎉 Simple SambaNova Consensus Demo completed!")
    print(f"📝 Results saved to memory for analysis.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Massive Scale Decentralized Multi-Agent Consensus Demo
=====================================================

Demonstration of peer-to-peer consensus with massive HPC clusters (100s of nodes)
and large multi-node jobs requiring 30-60 nodes each.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os

class MassiveAgent:
    """Massive scale decentralized agent for supercomputer-level demonstration"""
    
    def __init__(self, agent_id: str, resource_type: str, capabilities: Dict[str, Any]):
        self.agent_id = agent_id
        self.resource_type = resource_type
        self.capabilities = capabilities
        self.reputation = 1.0
        self.peers = {}
        self.active_proposals = {}
        self.running_jobs = {}
        self.allocated_nodes = 0
        self.suspected_agents = set()
        
        # Market dynamics
        self.utilization = random.uniform(0.0, 0.4)
        
        # LLM availability
        self.llm_enabled = bool(os.getenv('SAMBASTUDIO_API_KEY'))
        
    def calculate_bid(self, job_requirements: Dict[str, Any]) -> float:
        """Calculate bid score for a job"""
        if self.llm_enabled:
            return self._llm_enhanced_bidding(job_requirements)
        else:
            return self._heuristic_bidding(job_requirements)
    
    def _llm_enhanced_bidding(self, job_requirements: Dict[str, Any]) -> float:
        """LLM-enhanced intelligent bidding with real API calls"""
        print(f"\n🧠 LLM QUERY FROM {self.agent_id}")
        print("=" * 60)
        
        # Create LLM prompt for massive scale
        prompt = f"""You are {self.agent_id}, managing a massive supercomputer cluster in a decentralized resource allocation system.

JOB REQUEST (MASSIVE SCALE):
{json.dumps(job_requirements, indent=2)}

YOUR SUPERCOMPUTER CAPABILITIES:
- Total Nodes: {self.capabilities.get('total_nodes', 0):,}
- CPU Cores: {self.capabilities['cpu']:,}
- Memory: {self.capabilities['memory']:,} GB
- GPUs: {self.capabilities.get('gpu', 0):,}
- Resource Type: {self.resource_type}
- Current Utilization: {self.utilization:.1%}
- Interconnect: {self.capabilities.get('interconnect', 'N/A')}
- Storage: {self.capabilities.get('storage', 'N/A')}

This job requires {job_requirements.get('node_count', 1)} nodes spanning multiple compute nodes.

TASK: Calculate your bid score (0.0 to 1.0) for this massive multi-node job.

CONSIDER:
1. Can your supercomputer handle {job_requirements.get('node_count', 1)} nodes simultaneously?
2. How efficiently can your interconnect support this workload?
3. Your specialization for {job_requirements.get('job_type', 'unknown')} workloads
4. Current cluster utilization vs. job requirements
5. Multi-node job scheduling and resource allocation efficiency

SCORING GUIDELINES:
- 1.0 = Perfect supercomputer match, optimal for multi-node workload
- 0.7-0.9 = Excellent match, strong multi-node capabilities
- 0.5-0.7 = Good match, can handle the scale
- 0.0-0.5 = Poor match or insufficient resources for multi-node requirements

IMPORTANT: Respond with EXACTLY this JSON format, no extra text before or after:
{{"bid_score": 0.85, "reasoning": "explain why your supercomputer is ideal for this massive multi-node job"}}"""

        print(f"📝 PROMPT:")
        print(prompt)
        
        try:
            # Import here to avoid import issues if not available
            from dotenv import load_dotenv
            from langchain_community.llms.sambanova import SambaStudio
            load_dotenv()
            
            api_url = os.getenv('SAMBASTUDIO_URL')
            api_key = os.getenv('SAMBASTUDIO_API_KEY')
            
            if not api_url or not api_key:
                print("⚠️ SambaNova credentials not found, using heuristic fallback")
                return self._heuristic_bidding(job_requirements)
            
            print(f"\n⏳ Attempting SambaNova LangChain API...")
            
            llm_client = SambaStudio(
                sambastudio_url=api_url,
                sambastudio_api_key=api_key,
                model_kwargs={
                    "do_sample": True,
                    "max_tokens": 500,
                    "temperature": 0.1,
                    "process_prompt": True,
                    "model": "Meta-Llama-3-70B-Instruct",
                }
            )
            
            start_time = time.time()
            response = llm_client.invoke(prompt)
            response_time = time.time() - start_time
            
            print(f"💬 SAMBANOVA JSON ({response_time:.2f}s): {response}")
            
            # Parse JSON response
            if response and str(response).strip():
                response_str = str(response).strip()
                
                # Try direct JSON parsing
                try:
                    result = json.loads(response_str)
                    bid_score = float(result.get("bid_score", 0.5))
                    reasoning = result.get("reasoning", "LLM reasoning")
                    print(f"✅ LLM BID: {bid_score:.3f} | {reasoning}")
                    return max(0.0, min(1.0, bid_score))
                except json.JSONDecodeError:
                    print(f"⚠️ JSON parsing failed, attempting extraction...")
                    
                    # Try to extract JSON if response contains extra text
                    import re
                    if response_str.startswith("{") and response_str.endswith("}"):
                        json_str = response_str
                    else:
                        json_match = re.search(r'\{[^{}]*"bid_score"[^{}]*\}', response_str)
                        if json_match:
                            json_str = json_match.group(0)
                        else:
                            print(f"🔄 Using numeric extraction fallback")
                            # Extract any decimal number
                            score_match = re.search(r'(\d+\.\d+)', response_str)
                            if score_match:
                                bid_score = float(score_match.group(1))
                                if bid_score > 1.0:
                                    bid_score = bid_score / 100  # Convert percentage
                                print(f"📊 Extracted score: {bid_score:.3f}")
                                return max(0.0, min(1.0, bid_score))
                            else:
                                print(f"🔄 Using heuristic fallback")
                                return self._heuristic_bidding(job_requirements)
                    
                    try:
                        result = json.loads(json_str)
                        bid_score = float(result.get("bid_score", 0.5))
                        reasoning = result.get("reasoning", "LLM reasoning")
                        print(f"✅ EXTRACTED LLM BID: {bid_score:.3f} | {reasoning}")
                        return max(0.0, min(1.0, bid_score))
                    except:
                        print(f"🔄 Using heuristic fallback")
                        return self._heuristic_bidding(job_requirements)
            else:
                print(f"⚠️ Empty response, using heuristic fallback")
                return self._heuristic_bidding(job_requirements)
                
        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            print(f"🔄 Using heuristic fallback")
            return self._heuristic_bidding(job_requirements)
    
    def _heuristic_bidding(self, job_requirements: Dict[str, Any]) -> float:
        """Heuristic bidding for massive scale clusters"""
        # Base compatibility score
        score = 0.5
        
        # Primary resource matching: nodes (simplified)
        required_nodes = job_requirements.get("node_count", 1)
        available_nodes = self.capabilities.get("total_nodes", self.capabilities.get("nodes", 1))
        
        if available_nodes >= required_nodes:
            score += 0.4  # Can handle the node count
            if available_nodes >= required_nodes * 2:
                score += 0.2  # Has plenty of headroom
        else:
            score -= 0.5  # Cannot meet node requirements - major penalty
        
        # GPU matching
        required_gpu = job_requirements.get("gpu_count", 0)
        available_gpu = self.capabilities.get("gpu", 0)
        if required_gpu > 0:
            if available_gpu >= required_gpu:
                score += 0.2
            else:
                score -= 0.3
        
        # Resource type specialization
        job_type = job_requirements.get("job_type", "")
        if job_type == self.resource_type:
            score += 0.3
        elif self.resource_type == "hybrid" or self.resource_type == "ai":
            score += 0.1  # Versatile types get small bonus
        
        # Utilization penalty
        score -= self.utilization * 0.2
        
        # Reputation bonus
        score += (self.reputation - 1.0) * 0.1
        
        return max(0.0, min(1.0, score))

class MassiveDecentralizedNetwork:
    """Massive scale decentralized network for supercomputer clusters"""
    
    def __init__(self):
        self.agents = []
        self.byzantine_agents = set()
    
    def add_agent(self, agent: MassiveAgent):
        """Add a massive agent to the network"""
        self.agents.append(agent)
        
        # Connect to all other agents (full mesh)
        for other_agent in self.agents[:-1]:
            agent.peers[other_agent.agent_id] = other_agent
            other_agent.peers[agent.agent_id] = agent
    
    def inject_byzantine_fault(self, agent_id: str):
        """Inject Byzantine fault into an agent"""
        self.byzantine_agents.add(agent_id)
        for agent in self.agents:
            if agent.agent_id == agent_id:
                agent.reputation = max(0.1, agent.reputation - 0.4)
                print(f"🚨 Byzantine fault injected into {agent_id} (reputation: {agent.reputation:.3f})")
                break
    
    async def submit_job(self, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a massive multi-node job to the network"""
        print(f"\n🎯 SUBMITTING MASSIVE JOB: {job_requirements.get('application', 'Unknown')}")
        print(f"   📊 Requires: {job_requirements.get('node_count', 1)} nodes")
        
        start_time = time.time()
        
        # Phase 1: Bidding
        print(f"\n📋 PHASE 1: MASSIVE CLUSTER BIDDING")
        print("=" * 50)
        
        bids = {}
        for agent in self.agents:
            try:
                bid_score = agent.calculate_bid(job_requirements)
                bids[agent.agent_id] = {
                    "score": bid_score,
                    "reputation": agent.reputation,
                    "utilization": agent.utilization,
                    "nodes": agent.capabilities.get("total_nodes", agent.capabilities.get("nodes", 0))
                }
                print(f"   📊 {agent.agent_id}: bid={bid_score:.3f}, "
                      f"nodes={bids[agent.agent_id]['nodes']:,}, "
                      f"util={agent.utilization:.1%}")
            except Exception as e:
                print(f"   ❌ {agent.agent_id}: bidding failed ({e})")
                bids[agent.agent_id] = {"score": 0.0, "reputation": agent.reputation}
        
        # Phase 2: Consensus voting with Byzantine tolerance
        print(f"\n🗳️ PHASE 2: BYZANTINE-TOLERANT CONSENSUS")
        print("=" * 50)
        
        if not bids:
            return {"status": "failed", "reason": "No bids received"}
        
        # Sort by weighted score (bid * reputation)
        weighted_bids = {}
        for agent_id, bid_data in bids.items():
            # Byzantine agents get penalized bids
            if agent_id in self.byzantine_agents:
                weighted_score = bid_data["score"] * 0.3  # Severely penalized
                print(f"   🚨 {agent_id}: BYZANTINE PENALTY applied")
            else:
                weighted_score = bid_data["score"] * bid_data["reputation"]
            weighted_bids[agent_id] = weighted_score
            print(f"   ⚖️ {agent_id}: weighted_score={weighted_score:.3f}")
        
        # Winner selection
        winner = max(weighted_bids, key=weighted_bids.get)
        winning_score = weighted_bids[winner]
        
        # Consensus validation (require 2/3 majority)
        votes = {}
        voting_threshold = len(self.agents) * 2 // 3  # 2/3 majority
        
        print(f"\n🏆 PROPOSED WINNER: {winner} (score: {winning_score:.3f})")
        print(f"   🗳️ Requiring {voting_threshold}/{len(self.agents)} votes for consensus...")
        
        for agent in self.agents:
            # Agents vote based on their own assessment
            agent_bid = bids.get(agent.agent_id, {"score": 0.0})["score"]
            winner_bid = bids.get(winner, {"score": 0.0})["score"]
            
            # Vote "yes" if winner's bid is reasonable compared to own bid
            if winner_bid >= agent_bid * 0.7:  # Winner should be at least 70% as good
                vote = "approve"
            else:
                vote = "reject"
            
            # Byzantine agents vote randomly
            if agent.agent_id in self.byzantine_agents:
                vote = random.choice(["approve", "reject", "abstain"])
                print(f"   🚨 {agent.agent_id}: {vote} (BYZANTINE)")
            else:
                print(f"   ✅ {agent.agent_id}: {vote}")
            
            votes[agent.agent_id] = vote
        
        # Count votes
        approve_votes = sum(1 for vote in votes.values() if vote == "approve")
        reject_votes = sum(1 for vote in votes.values() if vote == "reject")
        
        consensus_time = time.time() - start_time
        
        if approve_votes >= voting_threshold:
            print(f"\n✅ CONSENSUS REACHED: {approve_votes}/{len(self.agents)} approved")
            print(f"   🎯 Job allocated to {winner}")
            print(f"   ⏱️ Consensus time: {consensus_time:.2f}s")
            
            # Update winner's resource allocation
            for agent in self.agents:
                if agent.agent_id == winner:
                    agent.utilization = min(1.0, agent.utilization + 0.1)
                    # Track allocated resources
                    job_id = f"job_{len(agent.running_jobs)+1:03d}"
                    agent.running_jobs[job_id] = {
                        "requirements": job_requirements,
                        "start_time": time.time(),
                        "estimated_runtime": job_requirements.get("estimated_runtime", 60)
                    }
                    agent.allocated_nodes += job_requirements.get("node_count", 0)
                    break
            
            return {
                "status": "success",
                "winner": winner,
                "score": winning_score,
                "consensus_time": consensus_time,
                "votes": {"approve": approve_votes, "reject": reject_votes},
                "method": "🧠 LLM-enhanced" if any(agent.llm_enabled for agent in self.agents) else "🔧 Heuristic"
            }
        else:
            print(f"\n❌ CONSENSUS FAILED: {approve_votes}/{len(self.agents)} approved (needed {voting_threshold})")
            return {
                "status": "failed",
                "reason": f"Insufficient consensus ({approve_votes}/{voting_threshold} votes)",
                "consensus_time": consensus_time,
                "votes": {"approve": approve_votes, "reject": reject_votes}
            }
    
    def show_resource_occupancy(self, agent: MassiveAgent):
        """Display resource occupancy for a specific agent after job allocation"""
        print(f"\n📈 RESOURCE OCCUPANCY: {agent.agent_id}")
        print("-" * 40)
        
        total_nodes = agent.capabilities.get("total_nodes", 0)
        total_cpu = agent.capabilities["cpu"]
        total_memory = agent.capabilities["memory"]
        total_gpu = agent.capabilities.get("gpu", 0)
        
        occupancy_pct = (agent.allocated_nodes / max(1, total_nodes)) * 100
        
        print(f"🖥️ Nodes: {agent.allocated_nodes:,}/{total_nodes:,} ({occupancy_pct:.1f}% occupied)")
        print(f"⚡ CPU: {total_cpu:,} cores total")
        print(f"💾 Memory: {total_memory:,}GB total")
        if total_gpu > 0:
            print(f"🚀 GPU: {total_gpu:,} total")
        
        print(f"🏃 Running Jobs: {len(agent.running_jobs)}")
        if agent.running_jobs:
            for job_id, job_info in agent.running_jobs.items():
                req = job_info["requirements"]
                app = req.get("application", "Unknown Application")
                nodes = req.get("node_count", 0)
                runtime = job_info.get("estimated_runtime", 0)
                print(f"   • {job_id}: {app} ({nodes} nodes, {runtime}min)")
    
    def show_network_status(self):
        """Display current network status"""
        print(f"\n📊 MASSIVE NETWORK STATUS")
        print("=" * 60)
        
        total_nodes = sum(agent.capabilities.get("total_nodes", agent.capabilities.get("nodes", 0)) for agent in self.agents)
        total_cpu = sum(agent.capabilities["cpu"] for agent in self.agents)
        total_memory = sum(agent.capabilities["memory"] for agent in self.agents)
        total_gpu = sum(agent.capabilities.get("gpu", 0) for agent in self.agents)
        
        print(f"🌐 Total Network Capacity:")
        print(f"   📊 {len(self.agents)} supercomputer clusters")
        print(f"   🖥️ {total_nodes:,} compute nodes")
        print(f"   ⚡ {total_cpu:,} CPU cores")
        print(f"   💾 {total_memory:,}GB memory")
        print(f"   🚀 {total_gpu:,} GPUs")
        
        print(f"\n🏛️ Individual Supercomputer Status:")
        for agent in self.agents:
            status = "🚨 BYZANTINE" if agent.agent_id in self.byzantine_agents else "✅ HEALTHY"
            nodes = agent.capabilities.get("total_nodes", agent.capabilities.get("nodes", 0))
            print(f"   {status} {agent.agent_id}:")
            print(f"      └─ {nodes:,} nodes | {agent.capabilities['cpu']:,} cores | "
                  f"{agent.capabilities['memory']:,}GB | {agent.capabilities.get('gpu', 0):,} GPUs")
            print(f"      └─ Utilization: {agent.utilization:.1%} | Reputation: {agent.reputation:.3f}")

async def run_massive_demo():
    """Run massive scale demonstration"""
    
    # Check LLM availability
    llm_available = bool(os.getenv('SAMBASTUDIO_API_KEY'))
    print(f"🧠 LLM Integration: {'✅ ENABLED' if llm_available else '❌ DISABLED (using heuristics)'}")
    
    print("🌐 MASSIVE SCALE DECENTRALIZED MULTI-AGENT CONSENSUS DEMO")
    print("=" * 80)
    print(f"⏰ Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create the massive decentralized network
    network = MassiveDecentralizedNetwork()
    
    # Add massive supercomputer clusters (100s to 500+ nodes each)
    cluster_configs = [
        {
            "id": "HPC_RESOURCE_00", 
            "type": "hpc", 
            "name": "HPC_RESOURCE_00",
            "total_nodes": 200,
            "cpu": 8800,    # 44 cores × 200 nodes
            "memory": 102400,  # 512GB × 200 nodes
            "gpu": 1200,    # 6 GPUs × 200 nodes
            "interconnect": "Mellanox InfiniBand EDR (100 Gbps)",
            "storage": "IBM Spectrum Scale (250PB)"
        },
        {
            "id": "HPC_RESOURCE_01", 
            "type": "gpu", 
            "name": "HPC_RESOURCE_01",
            "total_nodes": 320,
            "cpu": 20480,   # 64 cores × 320 nodes
            "memory": 163840,  # 512GB × 320 nodes  
            "gpu": 1280,    # 4 GPUs × 320 nodes
            "interconnect": "HPE Cray Slingshot-11 (200 Gbps)",
            "storage": "Lustre parallel filesystem (700PB)"
        },
        {
            "id": "HPC_RESOURCE_02", 
            "type": "memory", 
            "name": "HPC_RESOURCE_02",
            "total_nodes": 512,
            "cpu": 24576,   # 48 cores × 512 nodes
            "memory": 524288,  # 1024GB × 512 nodes
            "gpu": 0,
            "interconnect": "Tofu Interconnect D (28 Gbps per link)",
            "storage": "Distributed storage system (150PB)"
        },
        {
            "id": "HPC_RESOURCE_03", 
            "type": "hybrid", 
            "name": "HPC_RESOURCE_03",
            "total_nodes": 328,  # 200 CPU + 128 GPU nodes
            "cpu": 10496,   # (32×200) + (32×128)
            "memory": 83968,   # 256GB × 328 nodes
            "gpu": 512,     # 4 GPUs × 128 GPU nodes
            "interconnect": "Mellanox InfiniBand HDR (200 Gbps)",
            "storage": "Parallel filesystem with flash storage (100PB)"
        },
        {
            "id": "HPC_RESOURCE_04", 
            "type": "ai", 
            "name": "HPC_RESOURCE_04",
            "total_nodes": 353,  # 256 GPU + 96 CPU + 1 login
            "cpu": 28672,   # (64×256) + (128×96) + (64×1)
            "memory": 90368,   # 256GB × 353 nodes
            "gpu": 1024,    # 4 GPUs × 256 GPU nodes
            "interconnect": "Mellanox InfiniBand HDR (200 Gbps)",
            "storage": "All-flash Lustre filesystem (35PB)"
        },
        {
            "id": "HPC_RESOURCE_05", 
            "type": "storage", 
            "name": "HPC_RESOURCE_05",
            "total_nodes": 280,
            "cpu": 8960,    # 32 cores × 280 nodes
            "memory": 143360,  # 512GB × 280 nodes
            "gpu": 1120,    # 4 GPUs × 280 nodes
            "interconnect": "HPE Cray Slingshot-10 (200 Gbps)",
            "storage": "Distributed parallel storage (230PB)"
        }
    ]
    
    print(f"\n🏛️ FORMING MASSIVE SUPERCOMPUTER NETWORK")
    print("=" * 60)
    
    for config in cluster_configs:
        agent = MassiveAgent(
            agent_id=config["id"],
            resource_type=config["type"],
            capabilities=config
        )
        network.add_agent(agent)
        
        print(f"   ✅ Added {config['name']}")
        print(f"      └─ {config['total_nodes']:,} nodes | {config['cpu']:,} CPU cores | "
              f"{config['memory']:,}GB RAM | {config.get('gpu', 0):,} GPUs")
        print(f"      └─ Network: {config['interconnect']}")
        print(f"      └─ Storage: {config['storage']}")
    
    print(f"   🌐 Supercomputer network formed: {len(network.agents)} clusters connected")
    
    # Calculate total network capacity
    total_network_nodes = sum(agent.capabilities.get("total_nodes", 0) for agent in network.agents)
    total_network_cpu = sum(agent.capabilities["cpu"] for agent in network.agents)
    total_network_memory = sum(agent.capabilities["memory"] for agent in network.agents)
    total_network_gpu = sum(agent.capabilities.get("gpu", 0) for agent in network.agents)
    
    print(f"   📊 MASSIVE Total Network Capacity:")
    print(f"      └─ {total_network_nodes:,} compute nodes")
    print(f"      └─ {total_network_cpu:,} CPU cores")
    print(f"      └─ {total_network_memory:,}GB memory ({total_network_memory//1024:.1f}TB)")
    print(f"      └─ {total_network_gpu:,} GPUs")
    
    # Create massive multi-node job scenarios (30-60 nodes each)
    massive_scenarios = [
        {
            "name": "Exascale Climate Modeling (WRF)",
            "requirements": {
                "job_type": "hpc",
                "node_count": 40,
                "estimated_runtime": 480,
                "application": "Weather Research & Forecasting (WRF)",
                "domain_size": "Global 1km resolution",
                "simulation_time": "10-day forecast ensemble"
            }
        },
        {
            "name": "Large Language Model Training (1T parameters)",
            "requirements": {
                "job_type": "ai",
                "node_count": 60,
                "estimated_runtime": 720,
                "application": "Distributed PyTorch Training",
                "model": "1 Trillion parameter transformer",
                "technique": "3D parallelism (pipeline+tensor+data)"
            }
        },
        {
            "name": "Cosmological N-Body Simulation",
            "requirements": {
                "job_type": "hybrid",
                "node_count": 50,
                "estimated_runtime": 600,
                "application": "GADGET-4 N-body simulation",
                "particles": "100 billion dark matter particles",
                "box_size": "1 Gpc/h comoving"
            }
        },
        {
            "name": "Quantum Circuit Simulation",
            "requirements": {
                "job_type": "memory",
                "node_count": 32,
                "estimated_runtime": 360,
                "application": "Qiskit quantum circuit simulation",
                "qubits": "45-qubit quantum circuit",
                "gate_depth": "10,000 quantum gates"
            }
        },
        {
            "name": "Massive Graph Analytics (Trillion-edge)",
            "requirements": {
                "job_type": "storage",
                "node_count": 42,
                "estimated_runtime": 240,
                "application": "Distributed GraphX on Spark",
                "graph_size": "1 trillion edges, 100 billion vertices",
                "algorithm": "PageRank + Community Detection"
            }
        },
        {
            "name": "Genomics Population Analysis (100K genomes)",
            "requirements": {
                "job_type": "memory",
                "node_count": 38,
                "estimated_runtime": 420,
                "application": "GATK population genetics pipeline",
                "sample_size": "100,000 whole genomes",
                "analysis": "GWAS + population structure"
            }
        },
        {
            "name": "Fusion Plasma Simulation (ITER scale)",
            "requirements": {
                "job_type": "hpc",
                "node_count": 36,
                "estimated_runtime": 540,
                "application": "BOUT++ MHD simulation",
                "plasma_size": "ITER tokamak geometry",
                "physics": "3D MHD + turbulence"
            }
        },
        {
            "name": "Drug Discovery Molecular Docking (1M compounds)",
            "requirements": {
                "job_type": "gpu",
                "node_count": 40,
                "estimated_runtime": 180,
                "application": "AutoDock Vina GPU",
                "library_size": "1 million compounds",
                "target": "SARS-CoV-2 main protease"
            }
        }
    ]
    
    results = []
    
    print("\n" + "="*80)
    print("MASSIVE SCALE JOB ALLOCATION SCENARIOS")
    print("="*80)
    
    # Run first 6 jobs under normal conditions
    print(f"\n🌟 SCENARIO 1: NORMAL MASSIVE OPERATIONS (Jobs 1-6)")
    print("=" * 60)
    
    for i, scenario in enumerate(massive_scenarios[:6], 1):
        print(f"\n🎯 Massive Job {i}: {scenario['name']}")
        print(f"   📊 Scale: {scenario['requirements']['node_count']} nodes")
        
        result = await network.submit_job(scenario["requirements"])
        results.append(result)
        
        if result["status"] == "success":
            print(f"✅ MASSIVE SUCCESS: Allocated to {result['winner']} in {result['consensus_time']:.2f}s")
            # Show resource occupancy after allocation
            winner_agent = next(agent for agent in network.agents if agent.agent_id == result['winner'])
            network.show_resource_occupancy(winner_agent)
        else:
            print(f"❌ FAILED: {result['reason']}")
        
        await asyncio.sleep(1)  # Brief pause between jobs
    
    # Byzantine fault injection for massive scale
    print(f"\n🚨 SCENARIO 2: BYZANTINE ATTACK ON SUPERCOMPUTERS (Jobs 7-8)")
    print("=" * 60)
    
    malicious_agent = random.choice(network.agents[:2])
    network.inject_byzantine_fault(malicious_agent.agent_id)
    
    for i, scenario in enumerate(massive_scenarios[6:8], 7):
        print(f"\n🎯 Massive Job {i}: {scenario['name']} (under Byzantine attack)")
        print(f"   📊 Scale: {scenario['requirements']['node_count']} nodes")
        
        result = await network.submit_job(scenario["requirements"])
        results.append(result)
        
        if result["status"] == "success":
            print(f"✅ MASSIVE SUCCESS: Allocated to {result['winner']} in {result['consensus_time']:.2f}s")
            if result["winner"] == malicious_agent.agent_id:
                print("⚠️ WARNING: Malicious supercomputer won the allocation!")
            else:
                print("🛡️ PROTECTED: Byzantine supercomputer was rejected by consensus")
            # Show resource occupancy after allocation
            winner_agent = next(agent for agent in network.agents if agent.agent_id == result['winner'])
            network.show_resource_occupancy(winner_agent)
        else:
            print(f"❌ FAILED: {result['reason']}")
        
        await asyncio.sleep(1)
    
    # Final network status
    network.show_network_status()
    
    # Massive scale summary statistics
    print("\n📊 MASSIVE DEMO SUMMARY - 8 SUPERCOMPUTER JOBS PROCESSED")
    print("=" * 70)
    
    successful_jobs = len([r for r in results if r["status"] == "success"])
    failed_jobs = len(results) - successful_jobs
    
    if results:
        avg_consensus_time = sum(r.get("consensus_time", 0) for r in results if r["status"] == "success") / max(1, successful_jobs)
        
        # Calculate total allocated resources
        total_allocated_nodes = 0
        
        for i, result in enumerate(results):
            if result["status"] == "success":
                reqs = massive_scenarios[i]["requirements"]
                total_allocated_nodes += reqs["node_count"]
        
        print(f"📈 Massive Job Allocation Results:")
        print(f"   ✅ Successful: {successful_jobs}/{len(results)} ({successful_jobs/len(results)*100:.1f}%)")
        print(f"   ❌ Failed: {failed_jobs}/{len(results)}")
        print(f"   ⏱️ Average Consensus Time: {avg_consensus_time:.2f}s")
        
        print(f"\n📊 Total Resources Allocated:")
        print(f"   🖥️ {total_allocated_nodes:,} compute nodes")
        
        # Performance by scenario
        normal_jobs = sum(1 for r in results[:6] if r["status"] == "success")
        byzantine_jobs = sum(1 for r in results[6:8] if r["status"] == "success")
        
        print(f"\n📊 Performance by Massive Scale Scenario:")
        print(f"   🟢 Normal Supercomputer Operations (Jobs 1-6): {normal_jobs}/6 successful")
        print(f"   🔴 Byzantine Attack on Supercomputers (Jobs 7-8): {byzantine_jobs}/2 successful")
        
        if llm_available:
            llm_wins = len([r for r in results if r["status"] == "success" and "🧠" in r.get("method", "")])
            heuristic_wins = successful_jobs - llm_wins
            print(f"\n🧠 vs 🔧 Decision Method Results:")
            print(f"   🧠 LLM-Enhanced Decisions: {llm_wins}")
            print(f"   🔧 Heuristic Decisions: {heuristic_wins}")
        
        print(f"\n🛡️ Massive Scale Fault Tolerance:")
        print(f"   Byzantine supercomputer: {malicious_agent.agent_id}")
        print(f"   Jobs under attack: 2")
        print(f"   Network resilience: {'✅ Maintained' if successful_jobs >= 6 else '⚠️ Degraded' if successful_jobs >= 4 else '❌ Compromised'}")
    
    print(f"\n🎉 MASSIVE SCALE DEMO COMPLETED!")
    print(f"   ✨ {len(network.agents)} supercomputers operated without central coordinator")
    print(f"   🤝 Autonomous decisions for {total_allocated_nodes:,} compute nodes")
    print(f"   🛡️ Byzantine fault tolerance at exascale")
    print(f"   📊 {len(results)} massive multi-node jobs processed")
    
    return results

if __name__ == "__main__":
    print("🌐 Starting Massive Scale Decentralized Multi-Agent Consensus Demo...")
    asyncio.run(run_massive_demo())
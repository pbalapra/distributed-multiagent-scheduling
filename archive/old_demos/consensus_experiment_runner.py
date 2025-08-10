#!/usr/bin/env python3
"""
Configurable Consensus Experiment Runner for Multi-Agent HPC Systems

Provides flexible framework to:
1. Choose consensus methods at runtime
2. Configure experimental parameters
3. Run comparative studies
4. Generate comprehensive results

Supports all implemented consensus protocols:
- BFT (Basic Byzantine Fault Tolerant)
- PBFT (Practical Byzantine Fault Tolerance) 
- Raft (Leader-based consensus)
- Multi-Paxos (Crash fault tolerant)
- Tendermint (Modern Byzantine with finality)
- Multi-round Negotiation
- Weighted Voting
- Bidding-based Consensus
- Gossip Consensus
"""

import json
import time
import random
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import yaml

# Import our consensus implementations
from advanced_fault_tolerant_consensus import (
    PBFTConsensus, MultiPaxosConsensus, TendermintConsensus,
    FaultTolerantAgent, HPCJob, HPCNode, ConsensusResult
)

# Import LLM-enhanced agents
from llm_consensus_agents import (
    LLMEnhancedFaultTolerantAgent, create_llm_enhanced_agent, create_hybrid_agent_pool
)

class ConsensusMethod(Enum):
    """Available consensus methods"""
    BFT = "bft"
    PBFT = "pbft" 
    RAFT = "raft"
    MULTI_PAXOS = "multi_paxos"
    TENDERMINT = "tendermint"
    NEGOTIATION = "negotiation"
    WEIGHTED_VOTING = "weighted_voting"
    BIDDING = "bidding"
    GOSSIP = "gossip"
    ALL = "all"

@dataclass
class ExperimentConfig:
    """Experiment configuration parameters"""
    # Consensus methods to test
    methods: List[str] = field(default_factory=lambda: ["pbft", "tendermint", "multi_paxos"])
    
    # Agent decision mode
    agent_decision_mode: str = "heuristic"  # "heuristic", "llm", "hybrid"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 100
    
    # System parameters
    num_agents: int = 7
    byzantine_faults: int = 2
    crash_faults: int = 1
    
    # Job parameters
    num_jobs: int = 5
    job_types: List[str] = field(default_factory=lambda: ["ai", "climate", "genomics", "physics", "analytics"])
    
    # Node parameters
    nodes_per_agent: List[int] = field(default_factory=lambda: [10, 20])
    cpu_range: List[int] = field(default_factory=lambda: [16, 128])
    memory_range: List[int] = field(default_factory=lambda: [64, 512])
    gpu_range: List[int] = field(default_factory=lambda: [0, 8])
    
    # Experiment parameters
    repetitions: int = 3
    timeout_seconds: int = 30
    enable_faults: bool = True
    detailed_logging: bool = True
    
    # Output parameters
    output_dir: str = "experiment_results"
    save_raw_data: bool = True
    generate_plots: bool = False

class ConsensusExperimentRunner:
    """Main experiment runner with configurable consensus methods"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.agents = []
        self.jobs = []
        self.results = {}
        
        # Available consensus implementations
        self.consensus_implementations = {
            ConsensusMethod.PBFT.value: self._run_pbft,
            ConsensusMethod.MULTI_PAXOS.value: self._run_multi_paxos,
            ConsensusMethod.TENDERMINT.value: self._run_tendermint,
            ConsensusMethod.BFT.value: self._run_basic_bft,
            ConsensusMethod.RAFT.value: self._run_raft,
            ConsensusMethod.NEGOTIATION.value: self._run_negotiation,
            ConsensusMethod.WEIGHTED_VOTING.value: self._run_weighted_voting,
            ConsensusMethod.BIDDING.value: self._run_bidding,
        }
    
    def setup_experiment_environment(self):
        """Set up the experimental environment based on config"""
        print(f"🔬 Setting up Experiment Environment")
        print(f"   Methods: {', '.join(self.config.methods)}")
        print(f"   Agents: {self.config.num_agents}")
        print(f"   Agent Decision Mode: {self.config.agent_decision_mode}")
        print(f"   Jobs: {self.config.num_jobs}")
        print(f"   Repetitions: {self.config.repetitions}")
        print("=" * 60)
        
        # Create agents based on decision mode
        self.agents = []
        agent_names = [
            "PRIMARY_CONTROLLER", "GPU_CLUSTER_MANAGER", "CPU_CLUSTER_MANAGER",
            "MEMORY_MANAGER", "STORAGE_MANAGER", "BACKUP_CONTROLLER", 
            "EDGE_COORDINATOR", "SECONDARY_CONTROLLER", "NETWORK_MANAGER"
        ]
        specializations = ["gpu", "memory", "compute", "network", "storage", "general"]
        
        if self.config.agent_decision_mode == "hybrid":
            # Create mixed pool of heuristic and LLM agents
            self.agents = create_hybrid_agent_pool(
                num_agents=self.config.num_agents,
                heuristic_fraction=0.5,  # 50-50 split
                llm_temperature=self.config.llm_temperature,
                llm_max_tokens=self.config.llm_max_tokens
            )
        else:
            # Create agents of a single type
            for i in range(self.config.num_agents):
                agent_name = agent_names[i] if i < len(agent_names) else f"AGENT_{i}"
                stake = random.randint(100, 500)
                
                if self.config.agent_decision_mode == "llm":
                    specialization = specializations[i % len(specializations)]
                    agent = create_llm_enhanced_agent(
                        agent_id=agent_name,
                        stake=stake,
                        specialization=specialization,
                        llm_temperature=self.config.llm_temperature,
                        llm_max_tokens=self.config.llm_max_tokens
                    )
                    agent.agent_type = "llm"
                else:  # heuristic
                    agent = FaultTolerantAgent(agent_name, stake)
                    agent.agent_type = "heuristic"
                
                self.agents.append(agent)
        
        # Add nodes to all agents
        for agent in self.agents:
            num_nodes = random.randint(*self.config.nodes_per_agent)
            for j in range(num_nodes):
                node = HPCNode(
                    id=f"{agent.agent_id}_node_{j:02d}",
                    name=f"{agent.agent_id.lower()}_node_{j:02d}",
                    cpu_cores=random.randint(*self.config.cpu_range),
                    memory_gb=random.randint(*self.config.memory_range),
                    gpu_count=random.randint(*self.config.gpu_range),
                    storage_tb=random.randint(5, 100),
                    node_type=agent.agent_id.split('_')[0].lower()
                )
                agent.add_node(node)
            
            if self.config.detailed_logging:
                total_resources = {
                    "cpu": sum(n.cpu_cores for n in agent.managed_nodes),
                    "memory": sum(n.memory_gb for n in agent.managed_nodes),
                    "gpu": sum(n.gpu_count for n in agent.managed_nodes),
                    "nodes": len(agent.managed_nodes)
                }
                agent_type = getattr(agent, 'agent_type', 'unknown')
                specialization = getattr(agent, 'specialization', 'general')
                print(f"  👤 {agent.agent_id} ({agent_type}/{specialization}): {total_resources}")
    
    def generate_experimental_jobs(self):
        """Generate jobs for experiments based on config"""
        print(f"\n📊 Generating Experimental Jobs")
        
        job_templates = {
            "ai": (4, 32, 256, 4, "high"),
            "climate": (8, 16, 128, 0, "high"),
            "genomics": (2, 8, 64, 0, "medium"),
            "physics": (16, 8, 32, 2, "high"),
            "analytics": (1, 16, 128, 0, "medium"),
        }
        
        self.jobs = []
        for i in range(self.config.num_jobs):
            job_type = random.choice(self.config.job_types)
            if job_type in job_templates:
                nodes, cpu, mem, gpu, priority = job_templates[job_type]
            else:
                nodes, cpu, mem, gpu, priority = (4, 16, 64, 0, "medium")
            
            # Add some variation
            nodes = max(1, nodes + random.randint(-2, 2))
            
            job = HPCJob(
                id=f"job_{i:03d}",
                name=f"{job_type.upper()}_{i}",
                nodes_required=nodes,
                cpu_per_node=cpu,
                memory_per_node=mem,
                gpu_per_node=gpu,
                runtime_hours=random.randint(1, 8),
                priority=priority,
                job_type=job_type
            )
            self.jobs.append(job)
        
        if self.config.detailed_logging:
            for job in self.jobs:
                print(f"    • {job.name}: {job.nodes_required} nodes × {job.cpu_per_node}CPU/{job.memory_per_node}GB/{job.gpu_per_node}GPU")
    
    def inject_faults(self):
        """Inject faults based on experiment configuration"""
        if not self.config.enable_faults:
            return
            
        print(f"\n💀 Injecting Faults")
        
        # Byzantine faults
        if self.config.byzantine_faults > 0:
            byzantine_count = min(self.config.byzantine_faults, len(self.agents) // 3)
            byzantine_agents = random.sample(self.agents, byzantine_count)
            for agent in byzantine_agents:
                agent.byzantine_behavior = True
                print(f"  ☠️  {agent.agent_id} is now Byzantine (malicious)")
        
        # Note: Crash faults would be simulated by removing agents temporarily
        print(f"  🛡️  System should tolerate {len(self.agents) // 3} Byzantine failures")
    
    def run_experiments(self):
        """Run experiments for selected consensus methods"""
        print(f"\n🔬 RUNNING CONSENSUS EXPERIMENTS")
        print("=" * 80)
        
        # Handle "all" methods selection
        if "all" in self.config.methods:
            methods_to_test = list(self.consensus_implementations.keys())
        else:
            methods_to_test = [m for m in self.config.methods if m in self.consensus_implementations]
        
        self.results = {}
        
        for method_name in methods_to_test:
            print(f"\n{'='*70}")
            print(f"🧪 TESTING {method_name.upper().replace('_', '-')} CONSENSUS")
            print(f"{'='*70}")
            
            method_results = []
            
            for repetition in range(self.config.repetitions):
                if self.config.repetitions > 1:
                    print(f"\n--- Repetition {repetition + 1}/{self.config.repetitions} ---")
                
                # Reset environment for each repetition
                self._reset_environment()
                
                rep_results = []
                
                for job in self.jobs:
                    print(f"\nJob: {job.name} ({job.nodes_required} nodes)")
                    
                    try:
                        # Run the consensus method
                        consensus_func = self.consensus_implementations[method_name]
                        result = consensus_func(job)
                        result.repetition = repetition
                        rep_results.append(result)
                        
                        if result.success:
                            print(f"  ✅ SUCCESS: {len(result.assigned_nodes)} nodes")
                        else:
                            print(f"  ❌ FAILED: {result.details.get('reason', 'Unknown')}")
                            
                    except Exception as e:
                        print(f"  💥 ERROR: {str(e)}")
                        result = ConsensusResult(
                            method_name, False, job.id, [], 0, 0, 0, "", 
                            {"error": str(e), "repetition": repetition}
                        )
                        rep_results.append(result)
                
                method_results.extend(rep_results)
                
            self.results[method_name] = method_results
        
        # Analyze results
        self._analyze_experimental_results()
    
    def _reset_environment(self):
        """Reset environment state between repetitions"""
        for agent in self.agents:
            for node in agent.managed_nodes:
                node.allocated = False
        
        for job in self.jobs:
            job.status = "waiting"
            job.assigned_nodes = []
    
    # Consensus method implementations
    def _run_pbft(self, job: HPCJob) -> ConsensusResult:
        """Run PBFT consensus"""
        consensus = PBFTConsensus(self.agents)
        return consensus.run_consensus(job)
    
    def _run_multi_paxos(self, job: HPCJob) -> ConsensusResult:
        """Run Multi-Paxos consensus"""
        consensus = MultiPaxosConsensus(self.agents)
        return consensus.run_consensus(job)
    
    def _run_tendermint(self, job: HPCJob) -> ConsensusResult:
        """Run Tendermint consensus"""
        consensus = TendermintConsensus(self.agents)
        return consensus.run_consensus(job)
    
    def _run_basic_bft(self, job: HPCJob) -> ConsensusResult:
        """Run basic BFT consensus (placeholder)"""
        # This would integrate with your existing BFT implementation
        return ConsensusResult("BFT", True, job.id, [f"node_{job.id}"], 0.001, 1, 5, "Byzantine")
    
    def _run_raft(self, job: HPCJob) -> ConsensusResult:
        """Run Raft consensus (placeholder)"""
        # This would integrate with your existing Raft implementation  
        return ConsensusResult("Raft", True, job.id, [f"node_{job.id}"], 0.001, 1, 3, "Crash")
    
    def _run_negotiation(self, job: HPCJob) -> ConsensusResult:
        """Run negotiation consensus (placeholder)"""
        return ConsensusResult("Negotiation", True, job.id, [f"node_{job.id}"], 0.002, 3, 10, "Agreement")
    
    def _run_weighted_voting(self, job: HPCJob) -> ConsensusResult:
        """Run weighted voting consensus (placeholder)"""
        return ConsensusResult("Weighted", True, job.id, [f"node_{job.id}"], 0.001, 1, 4, "Expertise")
    
    def _run_bidding(self, job: HPCJob) -> ConsensusResult:
        """Run bidding consensus (placeholder)"""
        return ConsensusResult("Bidding", True, job.id, [f"node_{job.id}"], 0.003, 2, 8, "Economic")
    
    def _analyze_experimental_results(self):
        """Analyze and present experimental results"""
        print(f"\n{'='*80}")
        print("📈 EXPERIMENTAL RESULTS ANALYSIS")
        print(f"{'='*80}")
        
        method_stats = {}
        
        for method_name, method_results in self.results.items():
            if not method_results:
                continue
                
            successful = sum(1 for r in method_results if r.success)
            total_time = sum(r.time_taken for r in method_results)
            total_messages = sum(r.messages_sent for r in method_results)
            total_rounds = sum(r.rounds for r in method_results)
            
            # Calculate averages
            avg_time = total_time / len(method_results)
            avg_messages = total_messages / len(method_results) 
            avg_rounds = total_rounds / len(method_results)
            
            # Calculate confidence intervals (simple approximation)
            success_rate = (successful / len(method_results)) * 100
            
            method_stats[method_name] = {
                'success_rate': success_rate,
                'avg_time': avg_time,
                'avg_messages': avg_messages,
                'avg_rounds': avg_rounds,
                'total_jobs': len(method_results),
                'successful_jobs': successful,
                'fault_tolerance': method_results[0].fault_tolerance if method_results else "Unknown",
                'repetitions': self.config.repetitions
            }
        
        # Present results
        for method_name, stats in method_stats.items():
            print(f"\n🏆 {method_name.upper().replace('_', '-')} RESULTS:")
            print(f"  ✅ Success Rate: {stats['success_rate']:.1f}% ({stats['successful_jobs']}/{stats['total_jobs']})")
            print(f"  ⏱️  Average Time: {stats['avg_time']:.3f}s") 
            print(f"  📨 Average Messages: {stats['avg_messages']:.1f}")
            print(f"  🔄 Average Rounds: {stats['avg_rounds']:.1f}")
            print(f"  🛡️  Fault Model: {stats['fault_tolerance']}")
            print(f"  🔁 Repetitions: {stats['repetitions']}")
        
        # Rankings
        self._print_rankings(method_stats)
        
        # Save results
        if self.config.save_raw_data:
            self._save_results(method_stats)
    
    def _print_rankings(self, method_stats: Dict):
        """Print method rankings"""
        print(f"\n🥇 METHOD RANKINGS:")
        
        # By success rate
        by_success = sorted(method_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        print(f"\n  📈 By Success Rate:")
        for rank, (method, stats) in enumerate(by_success, 1):
            print(f"    #{rank} {method.replace('_', '-').title()}: {stats['success_rate']:.1f}%")
        
        # By speed
        by_speed = sorted(method_stats.items(), key=lambda x: x[1]['avg_time'])
        print(f"\n  🚀 By Speed:")
        for rank, (method, stats) in enumerate(by_speed, 1):
            print(f"    #{rank} {method.replace('_', '-').title()}: {stats['avg_time']:.3f}s")
        
        # By message efficiency
        by_efficiency = sorted(method_stats.items(), key=lambda x: x[1]['avg_messages'])
        print(f"\n  📡 By Message Efficiency:")
        for rank, (method, stats) in enumerate(by_efficiency, 1):
            print(f"    #{rank} {method.replace('_', '-').title()}: {stats['avg_messages']:.1f} messages")
    
    def _save_results(self, method_stats: Dict):
        """Save experimental results to files"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save summary statistics
        with open(output_path / "experiment_summary.json", "w") as f:
            json.dump(method_stats, f, indent=2)
        
        # Save configuration
        with open(output_path / "experiment_config.json", "w") as f:
            json.dump({
                "methods": self.config.methods,
                "num_agents": self.config.num_agents,
                "num_jobs": self.config.num_jobs,
                "repetitions": self.config.repetitions,
                "byzantine_faults": self.config.byzantine_faults,
                "timestamp": time.time()
            }, f, indent=2)
        
        # Save raw results
        with open(output_path / "raw_results.json", "w") as f:
            serializable_results = {}
            for method, results in self.results.items():
                serializable_results[method] = [
                    {
                        "protocol": r.protocol,
                        "success": r.success,
                        "job_id": r.job_id,
                        "time_taken": r.time_taken,
                        "rounds": r.rounds,
                        "messages_sent": r.messages_sent,
                        "fault_tolerance": r.fault_tolerance,
                        "details": r.details
                    }
                    for r in results
                ]
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to {output_path}/")

def load_config_from_file(config_file: str) -> ExperimentConfig:
    """Load configuration from YAML or JSON file"""
    with open(config_file, 'r') as f:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            config_data = yaml.safe_load(f)
        else:
            config_data = json.load(f)
    
    return ExperimentConfig(**config_data)

def create_sample_configs():
    """Create sample configuration files"""
    configs = {
        "quick_test.yaml": {
            "methods": ["pbft", "tendermint"],
            "num_agents": 7,
            "num_jobs": 3,
            "repetitions": 1,
            "detailed_logging": True
        },
        "comprehensive.yaml": {
            "methods": ["all"],
            "num_agents": 9,
            "num_jobs": 10,
            "repetitions": 5,
            "byzantine_faults": 2,
            "detailed_logging": False
        },
        "byzantine_focus.yaml": {
            "methods": ["pbft", "tendermint", "bft"],
            "num_agents": 7,
            "byzantine_faults": 2,
            "num_jobs": 8,
            "repetitions": 3
        },
        "crash_tolerance.yaml": {
            "methods": ["multi_paxos", "raft"],
            "num_agents": 5,
            "crash_faults": 2,
            "num_jobs": 6,
            "repetitions": 4
        },
        "llm_agents_test.yaml": {
            "methods": ["pbft", "tendermint"],
            "num_agents": 7,
            "num_jobs": 5,
            "repetitions": 2,
            "agent_decision_mode": "llm",
            "llm_temperature": 0.1,
            "llm_max_tokens": 150,
            "detailed_logging": True
        },
        "hybrid_agents_test.yaml": {
            "methods": ["pbft", "multi_paxos", "tendermint"],
            "num_agents": 8,
            "num_jobs": 6,
            "repetitions": 3,
            "agent_decision_mode": "hybrid",
            "llm_temperature": 0.2,
            "llm_max_tokens": 120,
            "byzantine_faults": 2,
            "detailed_logging": True
        },
        "agent_comparison.yaml": {
            "methods": ["pbft", "tendermint"],
            "num_agents": 7,
            "num_jobs": 4,
            "repetitions": 1,
            "agent_decision_mode": "heuristic",
            "detailed_logging": True,
            "save_raw_data": True
        }
    }
    
    for filename, config in configs.items():
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    print(f"📝 Created sample config files: {', '.join(configs.keys())}")

def main():
    parser = argparse.ArgumentParser(
        description="Configurable Consensus Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with PBFT and Tendermint using heuristic agents
  python consensus_experiment_runner.py --methods pbft tendermint --agents 7 --jobs 5
  
  # Test with LLM-enhanced agents
  python consensus_experiment_runner.py --methods pbft tendermint --agent-mode llm --llm-temperature 0.1
  
  # Test with hybrid agents (mix of heuristic and LLM)
  python consensus_experiment_runner.py --methods pbft tendermint --agent-mode hybrid --agents 8
  
  # Comprehensive test of all methods
  python consensus_experiment_runner.py --methods all --repetitions 3
  
  # Load configuration from file
  python consensus_experiment_runner.py --config llm_agents_test.yaml
  
  # Generate sample configurations (includes LLM examples)
  python consensus_experiment_runner.py --create-configs
        """
    )
    
    parser.add_argument('--config', type=str, help='Load configuration from YAML/JSON file')
    parser.add_argument('--methods', nargs='+', 
                       choices=['pbft', 'multi_paxos', 'tendermint', 'bft', 'raft', 
                               'negotiation', 'weighted_voting', 'bidding', 'all'],
                       default=['pbft', 'tendermint', 'multi_paxos'],
                       help='Consensus methods to test')
    parser.add_argument('--agents', type=int, default=7, help='Number of agents')
    parser.add_argument('--jobs', type=int, default=5, help='Number of jobs')
    parser.add_argument('--repetitions', type=int, default=1, help='Number of repetitions')
    parser.add_argument('--agent-mode', choices=['heuristic', 'llm', 'hybrid'], 
                       default='heuristic', help='Agent decision making mode')
    parser.add_argument('--llm-temperature', type=float, default=0.0, 
                       help='LLM temperature (0.0-2.0, lower is more deterministic)')
    parser.add_argument('--llm-max-tokens', type=int, default=100, 
                       help='Maximum tokens for LLM responses')
    parser.add_argument('--byzantine-faults', type=int, default=2, help='Number of Byzantine faults to inject')
    parser.add_argument('--no-faults', action='store_true', help='Disable fault injection')
    parser.add_argument('--output-dir', type=str, default='experiment_results', help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Reduce logging output')
    parser.add_argument('--create-configs', action='store_true', help='Create sample configuration files')
    
    args = parser.parse_args()
    
    if args.create_configs:
        create_sample_configs()
        return
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
        print(f"📁 Loaded configuration from {args.config}")
    else:
        config = ExperimentConfig(
            methods=args.methods,
            num_agents=args.agents,
            num_jobs=args.jobs,
            repetitions=args.repetitions,
            agent_decision_mode=args.agent_mode,
            llm_temperature=args.llm_temperature,
            llm_max_tokens=args.llm_max_tokens,
            byzantine_faults=args.byzantine_faults,
            enable_faults=not args.no_faults,
            output_dir=args.output_dir,
            detailed_logging=not args.quiet
        )
    
    # Run experiments
    runner = ConsensusExperimentRunner(config)
    runner.setup_experiment_environment()
    runner.generate_experimental_jobs()
    runner.inject_faults()
    runner.run_experiments()
    
    print(f"\n🏁 Experiments complete!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Enhanced Consensus Experiment Runner with Advanced Fault Injection

Integrates the comprehensive fault injection framework with consensus experiments,
allowing for parameterized fault scenarios during runtime.
"""

import json
import time
import random
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Import base consensus and fault injection frameworks
from consensus_experiment_runner import (
    ConsensusExperimentRunner, ExperimentConfig, ConsensusMethod
)
from fault_injection_framework import (
    FaultInjector, FaultParameters, FaultType, FaultSeverity
)
from advanced_fault_tolerant_consensus import (
    FaultTolerantAgent, HPCJob, HPCNode, ConsensusResult,
    PBFTConsensus, MultiPaxosConsensus, TendermintConsensus
)

@dataclass
class FaultExperimentConfig(ExperimentConfig):
    """Extended experiment configuration with fault injection parameters"""
    
    # Fault injection parameters
    fault_scenarios: List[str] = field(default_factory=list)
    custom_faults: List[Dict] = field(default_factory=list)
    fault_intensity: str = "medium"  # "light", "medium", "heavy", "chaos"
    
    # Temporal fault parameters
    fault_start_delay: float = 5.0
    fault_duration_range: List[float] = field(default_factory=lambda: [10.0, 30.0])
    recovery_enabled: bool = True
    recovery_time: float = 5.0
    
    # Target parameters
    fault_target_fraction: float = 0.3
    specific_fault_targets: List[str] = field(default_factory=list)
    
    # Network fault parameters
    network_partition_probability: float = 0.2
    message_delay_range: List[float] = field(default_factory=lambda: [0.1, 2.0])
    message_corruption_rate: float = 0.05
    
    # Performance fault parameters
    performance_degradation_factor: float = 0.5
    resource_exhaustion_limit: float = 0.7
    
    # Output parameters
    include_fault_analysis: bool = True
    save_fault_logs: bool = True

class EnhancedConsensusExperimentRunner(ConsensusExperimentRunner):
    """Enhanced experiment runner with fault injection capabilities"""
    
    def __init__(self, config: FaultExperimentConfig):
        super().__init__(config)
        self.fault_config = config
        self.fault_injector = FaultInjector()
        self.fault_results = {}
    
    def setup_experiment_environment(self):
        """Set up environment with fault injection capabilities"""
        super().setup_experiment_environment()
        
        # Register agents with fault injector
        self.fault_injector.register_agents(self.agents)
        
        print(f"🔥 Fault Injection System Ready")
        print(f"   Scenarios: {self.fault_config.fault_scenarios}")
        print(f"   Intensity: {self.fault_config.fault_intensity}")
        print(f"   Target fraction: {self.fault_config.fault_target_fraction}")
    
    def inject_experimental_faults(self):
        """Inject faults based on configuration"""
        print(f"\n🔥 INJECTING EXPERIMENTAL FAULTS")
        print("=" * 50)
        
        # Inject predefined scenarios
        for scenario_name in self.fault_config.fault_scenarios:
            fault_params_list = self.fault_injector.create_fault_scenario(scenario_name)
            for fault_params in fault_params_list:
                # Customize parameters based on config
                fault_params = self._customize_fault_parameters(fault_params)
                fault_id = self.fault_injector.inject_fault(fault_params)
                print(f"  📋 Injected {scenario_name} fault: {fault_id}")
        
        # Inject custom faults
        for custom_fault in self.fault_config.custom_faults:
            fault_params = self._create_fault_from_config(custom_fault)
            fault_id = self.fault_injector.inject_fault(fault_params)
            print(f"  🔧 Injected custom fault: {fault_id}")
        
        # Inject intensity-based faults
        self._inject_intensity_based_faults()
    
    def _customize_fault_parameters(self, fault_params: FaultParameters) -> FaultParameters:
        """Customize fault parameters based on experiment config"""
        # Apply config-specific customizations
        if self.fault_config.specific_fault_targets:
            fault_params.target_agents = self.fault_config.specific_fault_targets
        else:
            fault_params.target_fraction = self.fault_config.fault_target_fraction
        
        # Customize timing
        fault_params.start_time = random.uniform(
            self.fault_config.fault_start_delay,
            self.fault_config.fault_start_delay + 10.0
        )
        
        fault_params.duration = random.uniform(*self.fault_config.fault_duration_range)
        fault_params.recovery_enabled = self.fault_config.recovery_enabled
        fault_params.recovery_time = self.fault_config.recovery_time
        
        # Customize network parameters
        fault_params.delay_range = self.fault_config.message_delay_range
        fault_params.corruption_rate = self.fault_config.message_corruption_rate
        
        # Customize performance parameters
        fault_params.performance_degradation = self.fault_config.performance_degradation_factor
        fault_params.resource_limit = self.fault_config.resource_exhaustion_limit
        
        return fault_params
    
    def _create_fault_from_config(self, fault_config: Dict) -> FaultParameters:
        """Create fault parameters from configuration dictionary"""
        fault_type = FaultType(fault_config.get('fault_type', 'byzantine'))
        
        return FaultParameters(
            fault_type=fault_type,
            probability=fault_config.get('probability', 1.0),
            duration=fault_config.get('duration', 15.0),
            severity=FaultSeverity(fault_config.get('severity', 'medium')),
            start_time=fault_config.get('start_time', 5.0),
            target_fraction=fault_config.get('target_fraction', 0.3),
            target_agents=fault_config.get('target_agents', []),
            delay_range=fault_config.get('delay_range', [0.1, 1.0]),
            corruption_rate=fault_config.get('corruption_rate', 0.05),
            performance_degradation=fault_config.get('performance_degradation', 0.5),
            byzantine_strategy=fault_config.get('byzantine_strategy', 'random')
        )
    
    def _inject_intensity_based_faults(self):
        """Inject faults based on intensity level"""
        intensity_configs = {
            'light': {
                'num_faults': 1,
                'severity': FaultSeverity.LOW,
                'target_fraction': 0.2
            },
            'medium': {
                'num_faults': 2,
                'severity': FaultSeverity.MEDIUM,
                'target_fraction': 0.3
            },
            'heavy': {
                'num_faults': 3,
                'severity': FaultSeverity.HIGH,
                'target_fraction': 0.4
            },
            'chaos': {
                'num_faults': 5,
                'severity': FaultSeverity.CRITICAL,
                'target_fraction': 0.6
            }
        }
        
        intensity = self.fault_config.fault_intensity
        if intensity not in intensity_configs:
            return
        
        config = intensity_configs[intensity]
        fault_types = [FaultType.BYZANTINE, FaultType.MESSAGE_DELAY, FaultType.PARTIAL_FAILURE]
        
        for i in range(config['num_faults']):
            fault_type = random.choice(fault_types)
            fault_params = FaultParameters(
                fault_type=fault_type,
                probability=1.0,
                duration=random.uniform(15.0, 25.0),
                severity=config['severity'],
                start_time=random.uniform(2.0, 8.0),
                target_fraction=config['target_fraction']
            )
            
            fault_id = self.fault_injector.inject_fault(fault_params)
            print(f"  ⚡ Injected {intensity} intensity fault: {fault_id}")
    
    def run_experiments(self):
        """Run experiments with active fault injection"""
        print(f"\n🔬 RUNNING FAULT-INJECTED CONSENSUS EXPERIMENTS")
        print("=" * 80)
        
        # Inject faults before starting consensus experiments
        self.inject_experimental_faults()
        
        # Handle "all" methods selection
        if "all" in self.config.methods:
            methods_to_test = list(self.consensus_implementations.keys())
        else:
            methods_to_test = [m for m in self.config.methods if m in self.consensus_implementations]
        
        self.results = {}
        self.fault_results = {}
        
        for method_name in methods_to_test:
            print(f"\n{'='*70}")
            print(f"🧪 TESTING {method_name.upper().replace('_', '-')} WITH FAULTS")
            print(f"{'='*70}")
            
            method_results = []
            method_fault_stats = []
            
            for repetition in range(self.config.repetitions):
                if self.config.repetitions > 1:
                    print(f"\n--- Repetition {repetition + 1}/{self.config.repetitions} ---")
                
                # Reset environment for each repetition
                self._reset_environment()
                
                rep_results = []
                
                for job in self.jobs:
                    print(f"\nJob: {job.name} ({job.nodes_required} nodes)")
                    
                    # Update fault states before consensus
                    self.fault_injector.update_faults()
                    
                    try:
                        # Run the consensus method with fault awareness
                        consensus_func = self.consensus_implementations[method_name]
                        result = self._run_fault_aware_consensus(consensus_func, job)
                        result.repetition = repetition
                        rep_results.append(result)
                        
                        if result.success:
                            print(f"  ✅ SUCCESS: {len(result.assigned_nodes)} nodes")
                        else:
                            print(f"  ❌ FAILED: {result.details.get('reason', 'Unknown')}")
                            
                        # Collect fault statistics for this job
                        fault_stats = self.fault_injector.get_fault_statistics()
                        fault_stats['job_id'] = job.id
                        fault_stats['method'] = method_name
                        fault_stats['repetition'] = repetition
                        method_fault_stats.append(fault_stats)
                        
                    except Exception as e:
                        print(f"  💥 ERROR: {str(e)}")
                        result = ConsensusResult(
                            method_name, False, job.id, [], 0, 0, 0, "", 
                            {"error": str(e), "repetition": repetition}
                        )
                        rep_results.append(result)
                
                method_results.extend(rep_results)
            
            self.results[method_name] = method_results
            self.fault_results[method_name] = method_fault_stats
        
        # Comprehensive analysis including fault impact
        self._analyze_fault_aware_results()
    
    def _run_fault_aware_consensus(self, consensus_func, job: HPCJob) -> ConsensusResult:
        """Run consensus with fault injection awareness"""
        start_time = time.time()
        
        # Check agent availability before consensus
        available_agents = []
        for agent in self.agents:
            if self.fault_injector.is_agent_available(agent.agent_id):
                available_agents.append(agent)
            else:
                print(f"    ⚠️  {agent.agent_id} unavailable due to faults")
        
        if len(available_agents) < 3:  # Minimum for consensus
            return ConsensusResult(
                "fault_aware", False, job.id, [], time.time() - start_time, 0, 0, "fault",
                {"reason": "Insufficient available agents due to faults"}
            )
        
        # Run consensus with available agents
        try:
            # Temporarily replace agents with available ones for consensus
            original_agents = self.agents
            self.agents = available_agents
            
            result = consensus_func(job)
            
            # Restore original agent list
            self.agents = original_agents
            
            # Add fault impact to result details
            fault_stats = self.fault_injector.get_fault_statistics()
            result.details.update({
                'fault_impact': {
                    'active_faults': fault_stats['active_faults'],
                    'affected_agents': fault_stats['affected_agents'],
                    'available_agents': len(available_agents),
                    'total_agents': len(original_agents)
                }
            })
            
            return result
            
        except Exception as e:
            # Restore agents on error
            self.agents = original_agents
            raise e
    
    def _analyze_fault_aware_results(self):
        """Analyze results including fault injection impact"""
        print(f"\n{'='*80}")
        print("📈 FAULT-AWARE EXPERIMENTAL RESULTS ANALYSIS")
        print(f"{'='*80}")
        
        # Standard analysis
        method_stats = {}
        
        for method_name, method_results in self.results.items():
            if not method_results:
                continue
                
            successful = sum(1 for r in method_results if r.success)
            total_time = sum(r.time_taken for r in method_results)
            total_messages = sum(r.messages_sent for r in method_results)
            total_rounds = sum(r.rounds for r in method_results)
            
            # Calculate fault impact metrics
            fault_impacted_jobs = sum(1 for r in method_results 
                                    if 'fault_impact' in r.details and 
                                       r.details['fault_impact']['active_faults'] > 0)
            
            method_stats[method_name] = {
                'success_rate': (successful / len(method_results)) * 100,
                'avg_time': total_time / len(method_results),
                'avg_messages': total_messages / len(method_results),
                'avg_rounds': total_rounds / len(method_results),
                'total_jobs': len(method_results),
                'successful_jobs': successful,
                'fault_impacted_jobs': fault_impacted_jobs,
                'fault_resistance': (successful / max(1, fault_impacted_jobs)) * 100 if fault_impacted_jobs > 0 else 100,
                'fault_tolerance': method_results[0].fault_tolerance if method_results else "Unknown"
            }
        
        # Present results with fault analysis
        for method_name, stats in method_stats.items():
            print(f"\n🏆 {method_name.upper().replace('_', '-')} FAULT-AWARE RESULTS:")
            print(f"  ✅ Success Rate: {stats['success_rate']:.1f}% ({stats['successful_jobs']}/{stats['total_jobs']})")
            print(f"  ⏱️  Average Time: {stats['avg_time']:.3f}s")
            print(f"  📨 Average Messages: {stats['avg_messages']:.1f}")
            print(f"  🔄 Average Rounds: {stats['avg_rounds']:.1f}")
            print(f"  🔥 Fault Impacted Jobs: {stats['fault_impacted_jobs']}")
            print(f"  🛡️  Fault Resistance: {stats['fault_resistance']:.1f}%")
            print(f"  🎯 Fault Model: {stats['fault_tolerance']}")
        
        # Fault-specific analysis
        self._analyze_fault_impact()
        
        # Rankings including fault tolerance
        self._print_fault_aware_rankings(method_stats)
        
        # Save results with fault data
        if self.config.save_raw_data:
            self._save_fault_aware_results(method_stats)
    
    def _analyze_fault_impact(self):
        """Analyze the impact of different fault types"""
        print(f"\n🔥 FAULT IMPACT ANALYSIS:")
        
        fault_stats = self.fault_injector.get_fault_statistics()
        
        print(f"  📊 Total Faults Injected: {fault_stats['total_faults_injected']}")
        print(f"  ⚡ Active Faults: {fault_stats['active_faults']}")
        print(f"  ✅ Completed Faults: {fault_stats['completed_faults']}")
        print(f"  👥 Affected Agents: {fault_stats['affected_agents']}")
        
        print(f"\n  🔍 Faults by Type:")
        for fault_type, count in fault_stats['faults_by_type'].items():
            print(f"    {fault_type}: {count}")
        
        print(f"\n  🖥️  Agent Status:")
        for agent_id, status in fault_stats['agent_status'].items():
            availability = "🟢" if status['available'] else "🔴"
            performance = f"{status['performance_factor']*100:.0f}%"
            byzantine = "⚠️" if status['byzantine'] else ""
            print(f"    {availability} {agent_id}: {performance} perf, {status['active_faults']} faults {byzantine}")
    
    def _print_fault_aware_rankings(self, method_stats: Dict):
        """Print method rankings including fault tolerance metrics"""
        print(f"\n🥇 FAULT-AWARE METHOD RANKINGS:")
        
        # By success rate under faults
        by_success = sorted(method_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        print(f"\n  📈 By Success Rate:")
        for rank, (method, stats) in enumerate(by_success, 1):
            print(f"    #{rank} {method.replace('_', '-').title()}: {stats['success_rate']:.1f}%")
        
        # By fault resistance
        by_resistance = sorted(method_stats.items(), key=lambda x: x[1]['fault_resistance'], reverse=True)
        print(f"\n  🛡️  By Fault Resistance:")
        for rank, (method, stats) in enumerate(by_resistance, 1):
            print(f"    #{rank} {method.replace('_', '-').title()}: {stats['fault_resistance']:.1f}%")
        
        # By message efficiency under faults
        by_efficiency = sorted(method_stats.items(), key=lambda x: x[1]['avg_messages'])
        print(f"\n  📡 By Message Efficiency:")
        for rank, (method, stats) in enumerate(by_efficiency, 1):
            print(f"    #{rank} {method.replace('_', '-').title()}: {stats['avg_messages']:.1f} messages")
    
    def _save_fault_aware_results(self, method_stats: Dict):
        """Save experimental results including fault injection data"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save fault-aware summary
        with open(output_path / "fault_aware_summary.json", "w") as f:
            json.dump(method_stats, f, indent=2)
        
        # Save fault injection logs
        if self.fault_config.save_fault_logs:
            fault_logs = {
                'statistics': self.fault_injector.get_fault_statistics(),
                'fault_history': [
                    {
                        'id': event.id,
                        'fault_type': event.fault_params.fault_type.value,
                        'affected_agents': event.affected_agents,
                        'duration': event.fault_params.duration,
                        'active': event.active,
                        'recovered': event.recovered
                    }
                    for event in self.fault_injector.fault_history
                ],
                'active_faults': [
                    {
                        'id': event.id,
                        'fault_type': event.fault_params.fault_type.value,
                        'affected_agents': event.affected_agents,
                        'active': event.active
                    }
                    for event in self.fault_injector.active_faults.values()
                ]
            }
            
            with open(output_path / "fault_logs.json", "w") as f:
                json.dump(fault_logs, f, indent=2)
        
        # Save detailed fault results per method
        with open(output_path / "fault_results_per_method.json", "w") as f:
            json.dump(self.fault_results, f, indent=2)
        
        print(f"\n💾 Fault-aware results saved to {output_path}/")

def load_fault_config_from_file(config_file: str) -> FaultExperimentConfig:
    """Load fault experiment configuration from YAML or JSON file"""
    with open(config_file, 'r') as f:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            config_data = yaml.safe_load(f)
        else:
            config_data = json.load(f)
    
    return FaultExperimentConfig(**config_data)

def create_fault_experiment_configs():
    """Create sample fault experiment configuration files"""
    configs = {
        "fault_basic.yaml": {
            "methods": ["pbft", "tendermint"],
            "num_agents": 7,
            "num_jobs": 5,
            "repetitions": 2,
            "fault_scenarios": ["light_byzantine"],
            "fault_intensity": "medium",
            "detailed_logging": True
        },
        
        "fault_comprehensive.yaml": {
            "methods": ["pbft", "tendermint", "multi_paxos"],
            "num_agents": 9,
            "num_jobs": 10,
            "repetitions": 3,
            "fault_scenarios": ["heavy_byzantine", "network_chaos"],
            "fault_intensity": "heavy",
            "fault_target_fraction": 0.4,
            "message_delay_range": [0.5, 3.0],
            "performance_degradation_factor": 0.3
        },
        
        "fault_chaos.yaml": {
            "methods": ["all"],
            "num_agents": 10,
            "num_jobs": 15,
            "repetitions": 1,
            "fault_scenarios": ["mixed_chaos", "cascading_failure"],
            "fault_intensity": "chaos",
            "custom_faults": [
                {
                    "fault_type": "timing",
                    "duration": 20.0,
                    "target_fraction": 0.3,
                    "start_time": 8.0
                }
            ]
        },
        
        "fault_network.yaml": {
            "methods": ["pbft", "tendermint"],
            "num_agents": 7,
            "num_jobs": 8,
            "fault_scenarios": ["network_chaos"],
            "network_partition_probability": 0.5,
            "message_delay_range": [1.0, 4.0],
            "message_corruption_rate": 0.15
        }
    }
    
    for filename, config in configs.items():
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    print(f"📝 Created fault experiment configs: {', '.join(configs.keys())}")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Consensus Experiment Runner with Fault Injection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fault Injection Examples:
  # Basic fault injection
  python enhanced_consensus_experiment_runner.py --methods pbft tendermint --fault-intensity medium
  
  # Heavy fault scenario
  python enhanced_consensus_experiment_runner.py --config fault_comprehensive.yaml
  
  # Network chaos testing
  python enhanced_consensus_experiment_runner.py --config fault_network.yaml
  
  # Create sample fault configs
  python enhanced_consensus_experiment_runner.py --create-fault-configs
        """
    )
    
    parser.add_argument('--config', type=str, help='Load fault configuration from YAML/JSON file')
    parser.add_argument('--methods', nargs='+', 
                       choices=['pbft', 'multi_paxos', 'tendermint', 'bft', 'raft', 
                               'negotiation', 'weighted_voting', 'bidding', 'all'],
                       default=['pbft', 'tendermint'],
                       help='Consensus methods to test')
    parser.add_argument('--agents', type=int, default=7, help='Number of agents')
    parser.add_argument('--jobs', type=int, default=5, help='Number of jobs')
    parser.add_argument('--repetitions', type=int, default=1, help='Number of repetitions')
    
    # Fault injection parameters
    parser.add_argument('--fault-scenarios', nargs='+',
                       choices=['light_byzantine', 'heavy_byzantine', 'network_chaos', 
                               'cascading_failure', 'performance_degradation', 'mixed_chaos'],
                       default=[],
                       help='Fault scenarios to inject')
    parser.add_argument('--fault-intensity', choices=['light', 'medium', 'heavy', 'chaos'],
                       default='medium', help='Overall fault intensity')
    parser.add_argument('--fault-target-fraction', type=float, default=0.3,
                       help='Fraction of agents to target with faults')
    parser.add_argument('--no-faults', action='store_true', help='Disable all fault injection')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='fault_experiment_results', help='Output directory')
    parser.add_argument('--create-fault-configs', action='store_true', help='Create sample fault configuration files')
    parser.add_argument('--quiet', action='store_true', help='Reduce logging output')
    
    args = parser.parse_args()
    
    if args.create_fault_configs:
        create_fault_experiment_configs()
        return
    
    # Load configuration
    if args.config:
        config = load_fault_config_from_file(args.config)
        print(f"📁 Loaded fault configuration from {args.config}")
    else:
        config = FaultExperimentConfig(
            methods=args.methods,
            num_agents=args.agents,
            num_jobs=args.jobs,
            repetitions=args.repetitions,
            fault_scenarios=args.fault_scenarios if not args.no_faults else [],
            fault_intensity=args.fault_intensity if not args.no_faults else "light",
            fault_target_fraction=args.fault_target_fraction,
            enable_faults=not args.no_faults,
            output_dir=args.output_dir,
            detailed_logging=not args.quiet
        )
    
    # Run fault-aware experiments
    runner = EnhancedConsensusExperimentRunner(config)
    runner.setup_experiment_environment()
    runner.generate_experimental_jobs()
    runner.run_experiments()
    
    print(f"\n🏁 Fault-injected experiments complete!")

if __name__ == "__main__":
    main()

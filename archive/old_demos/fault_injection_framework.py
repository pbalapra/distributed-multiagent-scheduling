#!/usr/bin/env python3
"""
Advanced Fault Injection Framework for Multi-Agent HPC Consensus Systems

Provides comprehensive fault injection capabilities with full parameterization:
1. Byzantine faults (malicious behavior)
2. Crash failures (agent unavailability)
3. Network partitions (communication isolation)
4. Message delays (network latency)
5. Message corruption (data integrity issues)
6. Partial failures (degraded performance)
7. Recovery scenarios (self-healing)
8. Temporal faults (time-based activation)

Each fault type can be configured with probability, duration, severity, and recovery patterns.
"""

import json
import time
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import threading
import uuid
from collections import defaultdict

class FaultType(Enum):
    """Types of faults that can be injected"""
    BYZANTINE = "byzantine"              # Malicious behavior
    CRASH = "crash"                     # Complete agent failure
    NETWORK_PARTITION = "partition"     # Network isolation
    MESSAGE_DELAY = "delay"             # Communication latency
    MESSAGE_CORRUPTION = "corruption"   # Data integrity issues
    PARTIAL_FAILURE = "partial"         # Performance degradation
    INTERMITTENT = "intermittent"       # On/off failures
    CASCADE = "cascade"                 # Cascading failures
    RESOURCE_EXHAUSTION = "exhaustion"  # Resource depletion
    TIMING_ATTACK = "timing"            # Timing-based attacks

class FaultSeverity(Enum):
    """Severity levels for fault injection"""
    LOW = "low"         # Minor impact, easy recovery
    MEDIUM = "medium"   # Moderate impact, possible recovery
    HIGH = "high"       # Major impact, difficult recovery
    CRITICAL = "critical" # System-threatening

@dataclass
class FaultParameters:
    """Parameters for a specific fault injection"""
    fault_type: FaultType
    probability: float = 0.1        # Probability of activation (0.0 - 1.0)
    duration: float = 10.0          # Duration in seconds
    severity: FaultSeverity = FaultSeverity.MEDIUM
    
    # Timing parameters
    start_time: float = 0.0         # When to start (seconds after experiment start)
    end_time: Optional[float] = None # When to end (None = use duration)
    
    # Recovery parameters
    recovery_enabled: bool = True
    recovery_time: float = 5.0      # Time to recover after fault ends
    
    # Specific parameters per fault type
    target_agents: List[str] = field(default_factory=list)  # Specific agents to target
    target_fraction: float = 0.3    # Fraction of agents to affect if no specific targets
    
    # Network fault parameters
    partition_groups: List[List[str]] = field(default_factory=list)  # For network partitions
    delay_range: List[float] = field(default_factory=lambda: [0.1, 2.0])  # Min/max delay
    corruption_rate: float = 0.05   # Rate of message corruption
    
    # Performance parameters
    performance_degradation: float = 0.5  # Factor by which performance degrades
    resource_limit: float = 0.8     # Resource availability fraction
    
    # Byzantine behavior parameters
    byzantine_strategy: str = "random"  # "random", "selective", "coordinated"
    malicious_vote_rate: float = 0.3   # Rate of malicious voting

@dataclass
class FaultEvent:
    """A specific fault event instance"""
    id: str
    fault_params: FaultParameters
    affected_agents: List[str]
    start_time: float
    end_time: Optional[float] = None
    active: bool = False
    recovered: bool = False

class FaultInjector:
    """Main fault injection engine"""
    
    def __init__(self):
        self.active_faults: Dict[str, FaultEvent] = {}
        self.fault_history: List[FaultEvent] = []
        self.experiment_start_time = time.time()
        self.fault_statistics = defaultdict(int)
        
        # Agent state tracking
        self.agent_states = {}  # agent_id -> state info
        self.network_partitions = []
        self.message_delays = {}
        self.corrupted_messages = set()
        
    def register_agents(self, agents: List[Any]):
        """Register agents for fault injection"""
        for agent in agents:
            self.agent_states[agent.agent_id] = {
                'agent': agent,
                'original_state': self._capture_agent_state(agent),
                'current_faults': [],
                'performance_factor': 1.0,
                'network_accessible': True,
                'last_heartbeat': time.time()
            }
    
    def _capture_agent_state(self, agent) -> Dict:
        """Capture original agent state for recovery"""
        return {
            'byzantine_behavior': getattr(agent, 'byzantine_behavior', False),
            'performance_factor': 1.0,
            'network_delay': 0.0
        }
    
    def inject_fault(self, fault_params: FaultParameters) -> str:
        """Inject a fault based on parameters"""
        fault_id = str(uuid.uuid4())[:8]
        
        # Determine target agents
        if fault_params.target_agents:
            target_agents = fault_params.target_agents
        else:
            all_agents = list(self.agent_states.keys())
            num_targets = max(1, int(len(all_agents) * fault_params.target_fraction))
            target_agents = random.sample(all_agents, num_targets)
        
        # Calculate timing
        current_time = time.time()
        actual_start_time = current_time + fault_params.start_time
        
        if fault_params.end_time:
            actual_end_time = current_time + fault_params.end_time
        else:
            actual_end_time = actual_start_time + fault_params.duration
        
        # Create fault event
        fault_event = FaultEvent(
            id=fault_id,
            fault_params=fault_params,
            affected_agents=target_agents,
            start_time=actual_start_time,
            end_time=actual_end_time
        )
        
        self.active_faults[fault_id] = fault_event
        self.fault_statistics[fault_params.fault_type.value] += 1
        
        print(f"🔥 Fault {fault_id} scheduled:")
        print(f"   Type: {fault_params.fault_type.value}")
        print(f"   Targets: {target_agents}")
        print(f"   Severity: {fault_params.severity.value}")
        print(f"   Duration: {fault_params.duration}s")
        
        return fault_id
    
    def update_faults(self):
        """Update fault states based on current time"""
        current_time = time.time()
        
        for fault_id, fault_event in list(self.active_faults.items()):
            # Activate fault if time has come
            if not fault_event.active and current_time >= fault_event.start_time:
                self._activate_fault(fault_event)
                fault_event.active = True
                print(f"⚡ Fault {fault_id} ACTIVATED")
            
            # Deactivate fault if duration expired
            elif fault_event.active and fault_event.end_time and current_time >= fault_event.end_time:
                self._deactivate_fault(fault_event)
                fault_event.active = False
                fault_event.recovered = fault_event.fault_params.recovery_enabled
                print(f"🔄 Fault {fault_id} DEACTIVATED")
                
                # Move to history
                self.fault_history.append(fault_event)
                del self.active_faults[fault_id]
    
    def _activate_fault(self, fault_event: FaultEvent):
        """Activate a specific fault"""
        fault_type = fault_event.fault_params.fault_type
        
        if fault_type == FaultType.BYZANTINE:
            self._activate_byzantine_fault(fault_event)
        elif fault_type == FaultType.CRASH:
            self._activate_crash_fault(fault_event)
        elif fault_type == FaultType.NETWORK_PARTITION:
            self._activate_network_partition(fault_event)
        elif fault_type == FaultType.MESSAGE_DELAY:
            self._activate_message_delay(fault_event)
        elif fault_type == FaultType.MESSAGE_CORRUPTION:
            self._activate_message_corruption(fault_event)
        elif fault_type == FaultType.PARTIAL_FAILURE:
            self._activate_partial_failure(fault_event)
        elif fault_type == FaultType.INTERMITTENT:
            self._activate_intermittent_fault(fault_event)
        elif fault_type == FaultType.CASCADE:
            self._activate_cascade_fault(fault_event)
        elif fault_type == FaultType.RESOURCE_EXHAUSTION:
            self._activate_resource_exhaustion(fault_event)
        elif fault_type == FaultType.TIMING_ATTACK:
            self._activate_timing_attack(fault_event)
    
    def _activate_byzantine_fault(self, fault_event: FaultEvent):
        """Activate Byzantine (malicious) behavior"""
        for agent_id in fault_event.affected_agents:
            if agent_id in self.agent_states:
                agent = self.agent_states[agent_id]['agent']
                agent.byzantine_behavior = True
                
                # Set specific Byzantine strategy
                strategy = fault_event.fault_params.byzantine_strategy
                if strategy == "selective":
                    # Target specific types of consensus
                    agent.byzantine_target_types = ["proposal", "vote"]
                elif strategy == "coordinated":
                    # Coordinate with other Byzantine agents
                    agent.byzantine_coordination = True
                
                self.agent_states[agent_id]['current_faults'].append(fault_event.id)
                print(f"  ☠️  {agent_id} turned Byzantine ({strategy})")
    
    def _activate_crash_fault(self, fault_event: FaultEvent):
        """Activate crash fault (agent unavailability)"""
        for agent_id in fault_event.affected_agents:
            if agent_id in self.agent_states:
                agent = self.agent_states[agent_id]['agent']
                
                # Mark agent as crashed
                self.agent_states[agent_id]['crashed'] = True
                self.agent_states[agent_id]['network_accessible'] = False
                self.agent_states[agent_id]['current_faults'].append(fault_event.id)
                
                print(f"  💀 {agent_id} CRASHED")
    
    def _activate_network_partition(self, fault_event: FaultEvent):
        """Activate network partition"""
        if fault_event.fault_params.partition_groups:
            partitions = fault_event.fault_params.partition_groups
        else:
            # Create random partitions
            agents = fault_event.affected_agents
            mid = len(agents) // 2
            partitions = [agents[:mid], agents[mid:]]
        
        self.network_partitions.append({
            'fault_id': fault_event.id,
            'partitions': partitions
        })
        
        print(f"  🌐 Network partition created: {partitions}")
    
    def _activate_message_delay(self, fault_event: FaultEvent):
        """Activate message delays"""
        delay_min, delay_max = fault_event.fault_params.delay_range
        
        for agent_id in fault_event.affected_agents:
            delay = random.uniform(delay_min, delay_max)
            self.message_delays[agent_id] = delay
            
            if agent_id in self.agent_states:
                self.agent_states[agent_id]['current_faults'].append(fault_event.id)
            
            print(f"  🐌 {agent_id} message delay: {delay:.2f}s")
    
    def _activate_message_corruption(self, fault_event: FaultEvent):
        """Activate message corruption"""
        corruption_rate = fault_event.fault_params.corruption_rate
        
        for agent_id in fault_event.affected_agents:
            if agent_id in self.agent_states:
                self.agent_states[agent_id]['corruption_rate'] = corruption_rate
                self.agent_states[agent_id]['current_faults'].append(fault_event.id)
            
            print(f"  🔧 {agent_id} message corruption: {corruption_rate*100:.1f}%")
    
    def _activate_partial_failure(self, fault_event: FaultEvent):
        """Activate partial failure (performance degradation)"""
        degradation = fault_event.fault_params.performance_degradation
        
        for agent_id in fault_event.affected_agents:
            if agent_id in self.agent_states:
                self.agent_states[agent_id]['performance_factor'] = degradation
                self.agent_states[agent_id]['current_faults'].append(fault_event.id)
            
            print(f"  📉 {agent_id} performance degraded to {degradation*100:.1f}%")
    
    def _activate_intermittent_fault(self, fault_event: FaultEvent):
        """Activate intermittent fault (on/off pattern)"""
        for agent_id in fault_event.affected_agents:
            if agent_id in self.agent_states:
                self.agent_states[agent_id]['intermittent'] = {
                    'fault_id': fault_event.id,
                    'on_duration': random.uniform(1, 3),
                    'off_duration': random.uniform(0.5, 2),
                    'last_toggle': time.time(),
                    'currently_on': True
                }
                self.agent_states[agent_id]['current_faults'].append(fault_event.id)
            
            print(f"  ⚡ {agent_id} intermittent fault activated")
    
    def _activate_cascade_fault(self, fault_event: FaultEvent):
        """Activate cascading fault (spreads to other agents)"""
        initial_agents = fault_event.affected_agents
        cascade_probability = 0.3
        
        for agent_id in initial_agents:
            if agent_id in self.agent_states:
                self.agent_states[agent_id]['cascade_source'] = True
                self.agent_states[agent_id]['cascade_probability'] = cascade_probability
                self.agent_states[agent_id]['current_faults'].append(fault_event.id)
            
            print(f"  🔗 {agent_id} cascade fault initiated")
    
    def _activate_resource_exhaustion(self, fault_event: FaultEvent):
        """Activate resource exhaustion"""
        resource_limit = fault_event.fault_params.resource_limit
        
        for agent_id in fault_event.affected_agents:
            if agent_id in self.agent_states:
                self.agent_states[agent_id]['resource_limit'] = resource_limit
                self.agent_states[agent_id]['current_faults'].append(fault_event.id)
            
            print(f"  🔋 {agent_id} resources limited to {resource_limit*100:.1f}%")
    
    def _activate_timing_attack(self, fault_event: FaultEvent):
        """Activate timing attack"""
        for agent_id in fault_event.affected_agents:
            if agent_id in self.agent_states:
                self.agent_states[agent_id]['timing_attack'] = {
                    'delay_factor': random.uniform(1.5, 3.0),
                    'selective': True
                }
                self.agent_states[agent_id]['current_faults'].append(fault_event.id)
            
            print(f"  ⏰ {agent_id} timing attack activated")
    
    def _deactivate_fault(self, fault_event: FaultEvent):
        """Deactivate a fault and potentially recover"""
        fault_type = fault_event.fault_params.fault_type
        
        for agent_id in fault_event.affected_agents:
            if agent_id in self.agent_states:
                # Remove fault from agent's active faults
                if fault_event.id in self.agent_states[agent_id]['current_faults']:
                    self.agent_states[agent_id]['current_faults'].remove(fault_event.id)
                
                # Recovery logic
                if fault_event.fault_params.recovery_enabled:
                    self._recover_agent(agent_id, fault_type, fault_event)
    
    def _recover_agent(self, agent_id: str, fault_type: FaultType, fault_event: FaultEvent):
        """Recover an agent from a specific fault type"""
        if agent_id not in self.agent_states:
            return
        
        agent_state = self.agent_states[agent_id]
        original_state = agent_state['original_state']
        
        if fault_type == FaultType.BYZANTINE:
            agent_state['agent'].byzantine_behavior = original_state['byzantine_behavior']
            print(f"  🔄 {agent_id} recovered from Byzantine fault")
        
        elif fault_type == FaultType.CRASH:
            agent_state['crashed'] = False
            agent_state['network_accessible'] = True
            print(f"  🔄 {agent_id} recovered from crash")
        
        elif fault_type == FaultType.MESSAGE_DELAY:
            if agent_id in self.message_delays:
                del self.message_delays[agent_id]
            print(f"  🔄 {agent_id} recovered from message delay")
        
        elif fault_type == FaultType.MESSAGE_CORRUPTION:
            agent_state['corruption_rate'] = 0.0
            print(f"  🔄 {agent_id} recovered from message corruption")
        
        elif fault_type == FaultType.PARTIAL_FAILURE:
            agent_state['performance_factor'] = 1.0
            print(f"  🔄 {agent_id} recovered from partial failure")
        
        elif fault_type == FaultType.RESOURCE_EXHAUSTION:
            agent_state['resource_limit'] = 1.0
            print(f"  🔄 {agent_id} recovered from resource exhaustion")
    
    def is_agent_available(self, agent_id: str) -> bool:
        """Check if agent is available for consensus"""
        if agent_id not in self.agent_states:
            return True
        
        state = self.agent_states[agent_id]
        
        # Check crash status
        if state.get('crashed', False):
            return False
        
        # Check network accessibility
        if not state.get('network_accessible', True):
            return False
        
        return True
    
    def should_corrupt_message(self, sender_id: str) -> bool:
        """Check if a message should be corrupted"""
        if sender_id not in self.agent_states:
            return False
        
        corruption_rate = self.agent_states[sender_id].get('corruption_rate', 0.0)
        return random.random() < corruption_rate
    
    def get_message_delay(self, sender_id: str) -> float:
        """Get message delay for an agent"""
        return self.message_delays.get(sender_id, 0.0)
    
    def get_performance_factor(self, agent_id: str) -> float:
        """Get performance degradation factor for an agent"""
        if agent_id not in self.agent_states:
            return 1.0
        
        return self.agent_states[agent_id].get('performance_factor', 1.0)
    
    def are_agents_partitioned(self, agent1_id: str, agent2_id: str) -> bool:
        """Check if two agents are in different network partitions"""
        for partition_info in self.network_partitions:
            partitions = partition_info['partitions']
            
            agent1_partition = None
            agent2_partition = None
            
            for i, partition in enumerate(partitions):
                if agent1_id in partition:
                    agent1_partition = i
                if agent2_id in partition:
                    agent2_partition = i
            
            if (agent1_partition is not None and agent2_partition is not None and 
                agent1_partition != agent2_partition):
                return True
        
        return False
    
    def get_fault_statistics(self) -> Dict:
        """Get comprehensive fault injection statistics"""
        active_count = len(self.active_faults)
        total_injected = len(self.fault_history) + active_count
        
        stats = {
            'total_faults_injected': total_injected,
            'active_faults': active_count,
            'completed_faults': len(self.fault_history),
            'faults_by_type': dict(self.fault_statistics),
            'affected_agents': len([a for a in self.agent_states.values() if a['current_faults']]),
            'current_time': time.time() - self.experiment_start_time
        }
        
        # Add detailed agent status
        agent_status = {}
        for agent_id, state in self.agent_states.items():
            agent_status[agent_id] = {
                'available': self.is_agent_available(agent_id),
                'performance_factor': state.get('performance_factor', 1.0),
                'active_faults': len(state['current_faults']),
                'byzantine': getattr(state['agent'], 'byzantine_behavior', False)
            }
        
        stats['agent_status'] = agent_status
        
        return stats
    
    def create_fault_scenario(self, scenario_name: str) -> List[FaultParameters]:
        """Create predefined fault scenarios for common testing"""
        scenarios = {
            'light_byzantine': [
                FaultParameters(
                    FaultType.BYZANTINE,
                    probability=1.0,
                    duration=20.0,
                    target_fraction=0.2,
                    severity=FaultSeverity.LOW
                )
            ],
            
            'heavy_byzantine': [
                FaultParameters(
                    FaultType.BYZANTINE,
                    probability=1.0,
                    duration=30.0,
                    target_fraction=0.4,
                    severity=FaultSeverity.HIGH,
                    byzantine_strategy="coordinated"
                )
            ],
            
            'network_chaos': [
                FaultParameters(
                    FaultType.NETWORK_PARTITION,
                    probability=1.0,
                    duration=15.0,
                    start_time=5.0
                ),
                FaultParameters(
                    FaultType.MESSAGE_DELAY,
                    probability=1.0,
                    duration=25.0,
                    target_fraction=0.5,
                    delay_range=[0.5, 2.0]
                ),
                FaultParameters(
                    FaultType.MESSAGE_CORRUPTION,
                    probability=1.0,
                    duration=20.0,
                    target_fraction=0.3,
                    corruption_rate=0.1
                )
            ],
            
            'cascading_failure': [
                FaultParameters(
                    FaultType.CASCADE,
                    probability=1.0,
                    duration=30.0,
                    start_time=10.0,
                    target_fraction=0.1,
                    severity=FaultSeverity.CRITICAL
                )
            ],
            
            'performance_degradation': [
                FaultParameters(
                    FaultType.PARTIAL_FAILURE,
                    probability=1.0,
                    duration=40.0,
                    target_fraction=0.6,
                    performance_degradation=0.3
                ),
                FaultParameters(
                    FaultType.RESOURCE_EXHAUSTION,
                    probability=1.0,
                    duration=25.0,
                    start_time=15.0,
                    target_fraction=0.3,
                    resource_limit=0.4
                )
            ],
            
            'mixed_chaos': [
                FaultParameters(FaultType.BYZANTINE, probability=1.0, duration=20.0, target_fraction=0.15),
                FaultParameters(FaultType.CRASH, probability=1.0, duration=15.0, start_time=5.0, target_fraction=0.1),
                FaultParameters(FaultType.MESSAGE_DELAY, probability=1.0, duration=30.0, target_fraction=0.4, delay_range=[0.2, 1.5]),
                FaultParameters(FaultType.INTERMITTENT, probability=1.0, duration=35.0, start_time=10.0, target_fraction=0.2)
            ]
        }
        
        return scenarios.get(scenario_name, [])

def create_fault_config_examples():
    """Create example fault configuration files"""
    configs = {
        'light_faults.json': {
            'faults': [
                {
                    'fault_type': 'byzantine',
                    'probability': 1.0,
                    'duration': 15.0,
                    'target_fraction': 0.2,
                    'severity': 'low'
                }
            ]
        },
        
        'heavy_faults.json': {
            'faults': [
                {
                    'fault_type': 'byzantine',
                    'probability': 1.0,
                    'duration': 25.0,
                    'target_fraction': 0.3,
                    'severity': 'high',
                    'byzantine_strategy': 'coordinated'
                },
                {
                    'fault_type': 'crash',
                    'probability': 1.0,
                    'duration': 20.0,
                    'start_time': 10.0,
                    'target_fraction': 0.15
                }
            ]
        },
        
        'network_faults.json': {
            'faults': [
                {
                    'fault_type': 'partition',
                    'probability': 1.0,
                    'duration': 18.0,
                    'start_time': 5.0
                },
                {
                    'fault_type': 'delay',
                    'probability': 1.0,
                    'duration': 30.0,
                    'target_fraction': 0.5,
                    'delay_range': [0.3, 2.0]
                }
            ]
        }
    }
    
    for filename, config in configs.items():
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"📝 Created fault config examples: {', '.join(configs.keys())}")

if __name__ == "__main__":
    # Example usage
    injector = FaultInjector()
    
    # Create example configurations
    create_fault_config_examples()
    
    # Example fault scenarios
    print("🔥 Available Fault Scenarios:")
    scenarios = ['light_byzantine', 'heavy_byzantine', 'network_chaos', 'cascading_failure', 'mixed_chaos']
    for scenario in scenarios:
        faults = injector.create_fault_scenario(scenario)
        print(f"  {scenario}: {len(faults)} fault types")
    
    print("\n✅ Fault injection framework ready!")

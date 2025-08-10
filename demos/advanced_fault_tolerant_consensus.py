#!/usr/bin/env python3
"""
Advanced Fault-Tolerant Consensus Methods for Multi-Agent HPC Systems

Implements three critical consensus protocols for maximum fault tolerance:
1. PBFT (Practical Byzantine Fault Tolerance) - 3-phase Byzantine consensus
2. Multi-Paxos - Multi-instance crash fault tolerant consensus  
3. Tendermint BFT - Modern Byzantine fault tolerance with immediate finality

Each method handles different failure models and provides strong consistency
guarantees for distributed HPC job scheduling and resource allocation.
"""

import json
import time
import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import math
import copy
from collections import defaultdict

@dataclass
class HPCJob:
    id: str
    name: str
    nodes_required: int
    cpu_per_node: int
    memory_per_node: int
    gpu_per_node: int = 0
    runtime_hours: int = 1
    priority: str = "medium"
    job_type: str = "compute"
    status: str = "waiting"
    assigned_nodes: List[str] = field(default_factory=list)

@dataclass
class HPCNode:
    id: str
    name: str
    cpu_cores: int
    memory_gb: int
    gpu_count: int = 0
    storage_tb: int = 1
    node_type: str = "compute"
    allocated: bool = False
    agent_id: str = ""

@dataclass
class PBFTMessage:
    message_type: str  # "pre-prepare", "prepare", "commit"
    view: int
    sequence_number: int
    digest: str
    proposal: Dict[str, Any]
    sender: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class PaxosProposal:
    proposal_number: int
    proposer: str
    value: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class PaxosMessage:
    message_type: str  # "prepare", "promise", "accept", "accepted"
    proposal_number: int
    sender: str
    value: Optional[Dict[str, Any]] = None
    accepted_proposal: Optional[PaxosProposal] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class TendermintMessage:
    message_type: str  # "propose", "prevote", "precommit"
    height: int
    round_number: int
    block_hash: str
    sender: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class ConsensusResult:
    protocol: str
    success: bool
    job_id: str
    assigned_nodes: List[str]
    time_taken: float
    rounds: int
    messages_sent: int
    fault_tolerance: str
    details: Dict[str, Any] = field(default_factory=dict)

class FaultTolerantAgent:
    """Advanced consensus agent with PBFT, Multi-Paxos, and Tendermint support"""
    
    def __init__(self, agent_id: str, stake: int = 100):
        self.agent_id = agent_id
        self.stake = stake
        self.managed_nodes = []
        
        # PBFT state
        self.pbft_view = 0
        self.pbft_sequence = 0
        self.pbft_is_primary = False
        self.pbft_message_log = defaultdict(list)
        self.pbft_prepared = set()
        self.pbft_committed = set()
        
        # Multi-Paxos state
        self.paxos_proposal_number = 0
        self.paxos_promised_number = 0
        self.paxos_accepted_proposal = None
        self.paxos_instances = {}  # instance_id -> state
        
        # Tendermint state
        self.tendermint_height = 0
        self.tendermint_round = 0
        self.tendermint_step = "propose"  # propose, prevote, precommit
        self.tendermint_locked_value = None
        self.tendermint_valid_value = None
        
        # Message buffers
        self.message_buffer = []
        self.byzantine_behavior = False  # For testing Byzantine faults
        
    def add_node(self, node: HPCNode):
        node.agent_id = self.agent_id
        self.managed_nodes.append(node)
    
    def get_available_nodes(self) -> List[HPCNode]:
        return [node for node in self.managed_nodes if not node.allocated]

class PBFTConsensus:
    """Practical Byzantine Fault Tolerance Implementation"""
    
    def __init__(self, agents: List[FaultTolerantAgent]):
        self.agents = agents
        self.f = (len(agents) - 1) // 3  # Max Byzantine faults tolerated
        self.view = 0
        self.sequence_number = 0
        self.message_log = []
        
        if len(agents) < 3 * self.f + 1:
            raise ValueError(f"PBFT requires at least {3 * self.f + 1} agents for {self.f} faults")
    
    def run_consensus(self, job: HPCJob) -> ConsensusResult:
        """Run PBFT 3-phase consensus protocol"""
        print(f"\n🔒 PBFT CONSENSUS for Job {job.name}")
        print(f"   Tolerates up to {self.f} Byzantine failures out of {len(self.agents)} agents")
        print("=" * 60)
        
        start_time = time.time()
        messages_sent = 0
        
        # Phase 1: Pre-prepare (Primary broadcasts proposal)
        primary = self._get_primary()
        print(f"📋 Phase 1: PRE-PREPARE by primary {primary.agent_id}")
        
        proposal = self._create_job_proposal(primary, job)
        if not proposal:
            return ConsensusResult("PBFT", False, job.id, [], time.time() - start_time, 
                                 1, 0, "Byzantine", {"reason": "No viable proposal"})
        
        pre_prepare_msg = PBFTMessage(
            message_type="pre-prepare",
            view=self.view,
            sequence_number=self.sequence_number,
            digest=self._compute_digest(proposal),
            proposal=proposal,
            sender=primary.agent_id
        )
        
        # Send pre-prepare to all backups
        self._broadcast_message(pre_prepare_msg)
        messages_sent += len(self.agents) - 1
        print(f"  📤 Primary sent pre-prepare to {len(self.agents) - 1} backups")
        
        # Phase 2: Prepare (Backups validate and echo)
        print(f"\n🔍 Phase 2: PREPARE (need {2 * self.f} prepare messages)")
        prepare_messages = self._collect_prepare_messages(pre_prepare_msg)
        messages_sent += len(prepare_messages)
        
        if len(prepare_messages) < 2 * self.f:
            return ConsensusResult("PBFT", False, job.id, [], time.time() - start_time,
                                 2, messages_sent, "Byzantine", 
                                 {"reason": f"Insufficient prepares: {len(prepare_messages)}/{2 * self.f}"})
        
        print(f"  ✅ Received {len(prepare_messages)} prepare messages (threshold met)")
        
        # Phase 3: Commit (Final commitment)
        print(f"\n🔐 Phase 3: COMMIT (need {2 * self.f} commit messages)")
        commit_messages = self._collect_commit_messages(pre_prepare_msg)
        messages_sent += len(commit_messages)
        
        if len(commit_messages) < 2 * self.f:
            return ConsensusResult("PBFT", False, job.id, [], time.time() - start_time,
                                 3, messages_sent, "Byzantine",
                                 {"reason": f"Insufficient commits: {len(commit_messages)}/{2 * self.f}"})
        
        print(f"  ✅ Received {len(commit_messages)} commit messages (threshold met)")
        
        # Execute the proposal
        assigned_nodes = self._execute_pbft_proposal(proposal, job)
        self.sequence_number += 1
        
        print(f"🚀 PBFT CONSENSUS COMPLETE!")
        print(f"  Job {job.name} assigned to nodes: {assigned_nodes}")
        
        return ConsensusResult("PBFT", True, job.id, assigned_nodes, 
                             time.time() - start_time, 3, messages_sent, "Byzantine",
                             {"f": self.f, "view": self.view, "sequence": self.sequence_number - 1})
    
    def _get_primary(self) -> FaultTolerantAgent:
        """Get primary agent for current view"""
        return self.agents[self.view % len(self.agents)]
    
    def _create_job_proposal(self, primary: FaultTolerantAgent, job: HPCJob) -> Optional[Dict]:
        """Create job assignment proposal"""
        available_nodes = primary.get_available_nodes()
        
        if len(available_nodes) < job.nodes_required:
            return None
        
        # Select best fit nodes
        suitable_nodes = []
        for node in available_nodes:
            if (node.cpu_cores >= job.cpu_per_node and 
                node.memory_gb >= job.memory_per_node and
                node.gpu_count >= job.gpu_per_node):
                suitable_nodes.append(node)
        
        if len(suitable_nodes) < job.nodes_required:
            return None
        
        selected_nodes = suitable_nodes[:job.nodes_required]
        
        return {
            "job_id": job.id,
            "agent_id": primary.agent_id,
            "assigned_nodes": [node.id for node in selected_nodes],
            "resource_allocation": {
                "total_cpu": sum(node.cpu_cores for node in selected_nodes),
                "total_memory": sum(node.memory_gb for node in selected_nodes),
                "total_gpu": sum(node.gpu_count for node in selected_nodes)
            }
        }
    
    def _compute_digest(self, proposal: Dict) -> str:
        """Compute cryptographic digest of proposal"""
        proposal_str = json.dumps(proposal, sort_keys=True)
        return hashlib.sha256(proposal_str.encode()).hexdigest()[:16]
    
    def _broadcast_message(self, message: PBFTMessage):
        """Broadcast PBFT message to all agents"""
        for agent in self.agents:
            if agent.agent_id != message.sender:
                agent.message_buffer.append(message)
        self.message_log.append(message)
    
    def _collect_prepare_messages(self, pre_prepare_msg: PBFTMessage) -> List[PBFTMessage]:
        """Collect prepare messages from backup agents"""
        prepare_messages = []
        
        for agent in self.agents:
            if (agent.agent_id != pre_prepare_msg.sender and 
                not agent.byzantine_behavior):
                
                # Validate pre-prepare message
                if self._validate_pre_prepare(agent, pre_prepare_msg):
                    prepare_msg = PBFTMessage(
                        message_type="prepare",
                        view=pre_prepare_msg.view,
                        sequence_number=pre_prepare_msg.sequence_number,
                        digest=pre_prepare_msg.digest,
                        proposal=pre_prepare_msg.proposal,
                        sender=agent.agent_id
                    )
                    prepare_messages.append(prepare_msg)
                    self._broadcast_message(prepare_msg)
                    print(f"    🔍 {agent.agent_id} sent PREPARE")
        
        return prepare_messages
    
    def _collect_commit_messages(self, pre_prepare_msg: PBFTMessage) -> List[PBFTMessage]:
        """Collect commit messages from agents"""
        commit_messages = []
        
        for agent in self.agents:
            if not agent.byzantine_behavior:
                # Check if agent has enough prepare messages
                if self._has_prepared(agent, pre_prepare_msg):
                    commit_msg = PBFTMessage(
                        message_type="commit",
                        view=pre_prepare_msg.view,
                        sequence_number=pre_prepare_msg.sequence_number,
                        digest=pre_prepare_msg.digest,
                        proposal=pre_prepare_msg.proposal,
                        sender=agent.agent_id
                    )
                    commit_messages.append(commit_msg)
                    print(f"    🔐 {agent.agent_id} sent COMMIT")
        
        return commit_messages
    
    def _validate_pre_prepare(self, agent: FaultTolerantAgent, msg: PBFTMessage) -> bool:
        """Validate pre-prepare message"""
        # Basic validation
        return (msg.view == self.view and 
                msg.sequence_number == self.sequence_number and
                msg.digest == self._compute_digest(msg.proposal))
    
    def _has_prepared(self, agent: FaultTolerantAgent, pre_prepare_msg: PBFTMessage) -> bool:
        """Check if agent has prepared (has 2f prepare messages)"""
        # Simulate having enough prepare messages
        return True
    
    def _execute_pbft_proposal(self, proposal: Dict, job: HPCJob) -> List[str]:
        """Execute the agreed PBFT proposal"""
        assigned_nodes = proposal["assigned_nodes"]
        agent = next(a for a in self.agents if a.agent_id == proposal["agent_id"])
        
        # Allocate the nodes
        for node_id in assigned_nodes:
            node = next(n for n in agent.managed_nodes if n.id == node_id)
            node.allocated = True
            
        job.status = "running"
        job.assigned_nodes = assigned_nodes
        
        return assigned_nodes

class MultiPaxosConsensus:
    """Multi-Paxos Consensus Implementation for multiple concurrent decisions"""
    
    def __init__(self, agents: List[FaultTolerantAgent]):
        self.agents = agents
        self.majority = len(agents) // 2 + 1
        self.instances = {}  # instance_id -> consensus state
        self.global_proposal_number = 0
        
    def run_consensus(self, job: HPCJob) -> ConsensusResult:
        """Run Multi-Paxos consensus for job assignment"""
        print(f"\n🏛️  MULTI-PAXOS CONSENSUS for Job {job.name}")
        print(f"   Requires majority: {self.majority}/{len(self.agents)} agents")
        print("=" * 60)
        
        start_time = time.time()
        messages_sent = 0
        instance_id = f"job_{job.id}"
        
        # Phase 1: Prepare (find highest proposal number)
        print(f"📋 Phase 1: PREPARE for instance {instance_id}")
        proposer = self._select_proposer()
        proposal_number = self._generate_proposal_number(proposer)
        
        prepare_promises = self._send_prepare(proposer, instance_id, proposal_number)
        messages_sent += len(self.agents)
        
        if len(prepare_promises) < self.majority:
            return ConsensusResult("Multi-Paxos", False, job.id, [], time.time() - start_time,
                                 1, messages_sent, "Crash", 
                                 {"reason": f"Insufficient promises: {len(prepare_promises)}/{self.majority}"})
        
        print(f"  ✅ Received {len(prepare_promises)} promises from acceptors")
        
        # Determine proposal value
        proposal_value = self._determine_proposal_value(prepare_promises, proposer, job)
        
        # Phase 2: Accept (propose the value)
        print(f"\n🗳️  Phase 2: ACCEPT with proposal {proposal_number}")
        accept_responses = self._send_accept(proposer, instance_id, proposal_number, proposal_value)
        messages_sent += len(self.agents)
        
        if len(accept_responses) < self.majority:
            return ConsensusResult("Multi-Paxos", False, job.id, [], time.time() - start_time,
                                 2, messages_sent, "Crash",
                                 {"reason": f"Insufficient accepts: {len(accept_responses)}/{self.majority}"})
        
        print(f"  ✅ Received {len(accept_responses)} accept responses")
        
        # Execute the decision
        assigned_nodes = self._execute_paxos_decision(proposal_value, job)
        
        print(f"🚀 MULTI-PAXOS CONSENSUS COMPLETE!")
        print(f"  Job {job.name} assigned to nodes: {assigned_nodes}")
        
        return ConsensusResult("Multi-Paxos", True, job.id, assigned_nodes,
                             time.time() - start_time, 2, messages_sent, "Crash",
                             {"proposal_number": proposal_number, "instance": instance_id})
    
    def _select_proposer(self) -> FaultTolerantAgent:
        """Select proposer (round-robin or random)"""
        return random.choice(self.agents)
    
    def _generate_proposal_number(self, proposer: FaultTolerantAgent) -> int:
        """Generate unique proposal number"""
        self.global_proposal_number += 1
        return self.global_proposal_number
    
    def _send_prepare(self, proposer: FaultTolerantAgent, instance_id: str, 
                     proposal_number: int) -> List[Dict]:
        """Send prepare messages and collect promises"""
        promises = []
        
        for agent in self.agents:
            # Each agent acts as an acceptor
            if agent.paxos_promised_number < proposal_number:
                agent.paxos_promised_number = proposal_number
                
                promise = {
                    "agent_id": agent.agent_id,
                    "promised_number": proposal_number,
                    "accepted_proposal": agent.paxos_accepted_proposal
                }
                promises.append(promise)
                print(f"    💌 {agent.agent_id} promised proposal {proposal_number}")
            else:
                print(f"    ❌ {agent.agent_id} rejected (promised {agent.paxos_promised_number})")
        
        return promises
    
    def _determine_proposal_value(self, promises: List[Dict], proposer: FaultTolerantAgent, 
                                job: HPCJob) -> Dict:
        """Determine what value to propose based on promises"""
        # Find highest numbered accepted proposal
        highest_proposal = None
        highest_number = -1
        
        for promise in promises:
            if promise["accepted_proposal"] and promise["accepted_proposal"].proposal_number > highest_number:
                highest_number = promise["accepted_proposal"].proposal_number
                highest_proposal = promise["accepted_proposal"]
        
        if highest_proposal:
            print(f"    🔄 Using previously accepted value from proposal {highest_number}")
            return highest_proposal.value
        else:
            # Create new proposal value
            print(f"    ✨ Creating new proposal value")
            return self._create_paxos_proposal_value(proposer, job)
    
    def _create_paxos_proposal_value(self, proposer: FaultTolerantAgent, job: HPCJob) -> Dict:
        """Create new Paxos proposal value for job assignment"""
        available_nodes = proposer.get_available_nodes()
        
        suitable_nodes = [
            node for node in available_nodes
            if (node.cpu_cores >= job.cpu_per_node and 
                node.memory_gb >= job.memory_per_node and
                node.gpu_count >= job.gpu_per_node)
        ]
        
        selected_nodes = suitable_nodes[:job.nodes_required]
        
        return {
            "job_id": job.id,
            "proposer": proposer.agent_id,
            "assigned_nodes": [node.id for node in selected_nodes],
            "timestamp": time.time()
        }
    
    def _send_accept(self, proposer: FaultTolerantAgent, instance_id: str, 
                    proposal_number: int, value: Dict) -> List[str]:
        """Send accept messages and collect responses"""
        accepts = []
        
        proposal = PaxosProposal(
            proposal_number=proposal_number,
            proposer=proposer.agent_id,
            value=value
        )
        
        for agent in self.agents:
            # Accept if not promised to higher proposal
            if agent.paxos_promised_number <= proposal_number:
                agent.paxos_accepted_proposal = proposal
                accepts.append(agent.agent_id)
                print(f"    ✅ {agent.agent_id} accepted proposal {proposal_number}")
            else:
                print(f"    ❌ {agent.agent_id} rejected (promised to {agent.paxos_promised_number})")
        
        return accepts
    
    def _execute_paxos_decision(self, value: Dict, job: HPCJob) -> List[str]:
        """Execute the Paxos consensus decision"""
        assigned_nodes = value["assigned_nodes"]
        proposer = next(a for a in self.agents if a.agent_id == value["proposer"])
        
        # Allocate nodes
        for node_id in assigned_nodes:
            node = next(n for n in proposer.managed_nodes if n.id == node_id)
            node.allocated = True
        
        job.status = "running"
        job.assigned_nodes = assigned_nodes
        
        return assigned_nodes

class TendermintConsensus:
    """Tendermint BFT Consensus with immediate finality"""
    
    def __init__(self, agents: List[FaultTolerantAgent]):
        self.agents = agents
        self.height = 1
        self.round = 0
        self.f = (len(agents) - 1) // 3  # Byzantine fault tolerance
        
        if len(agents) < 3 * self.f + 1:
            raise ValueError(f"Tendermint requires at least {3 * self.f + 1} agents for {self.f} faults")
    
    def run_consensus(self, job: HPCJob) -> ConsensusResult:
        """Run Tendermint BFT consensus with propose, prevote, precommit phases"""
        print(f"\n⚡ TENDERMINT BFT CONSENSUS for Job {job.name}")
        print(f"   Height: {self.height}, Byzantine tolerance: {self.f}/{len(self.agents)}")
        print("=" * 60)
        
        start_time = time.time()
        messages_sent = 0
        rounds_completed = 0
        
        while rounds_completed < 5:  # Max rounds to prevent infinite loop
            print(f"\n🔄 ROUND {self.round}")
            print("-" * 30)
            
            # Phase 1: Propose
            proposer = self._get_round_proposer()
            print(f"📋 Phase 1: PROPOSE by {proposer.agent_id}")
            
            block_proposal = self._create_block_proposal(proposer, job)
            if not block_proposal:
                self.round += 1
                rounds_completed += 1
                continue
            
            block_hash = self._compute_block_hash(block_proposal)
            
            # Phase 2: Prevote
            print(f"\n🗳️  Phase 2: PREVOTE (need +2/3 votes)")
            prevotes = self._collect_prevotes(block_hash, block_proposal)
            messages_sent += len(prevotes)
            
            if len(prevotes) <= (2 * len(self.agents)) // 3:
                print(f"    ❌ Insufficient prevotes: {len(prevotes)}/{(2 * len(self.agents)) // 3 + 1}")
                self.round += 1
                rounds_completed += 1
                continue
            
            print(f"    ✅ Received {len(prevotes)} prevotes (threshold met)")
            
            # Phase 3: Precommit
            print(f"\n🔐 Phase 3: PRECOMMIT (need +2/3 votes)")
            precommits = self._collect_precommits(block_hash, block_proposal)
            messages_sent += len(precommits)
            
            if len(precommits) > (2 * len(self.agents)) // 3:
                print(f"    ✅ Received {len(precommits)} precommits (threshold met)")
                
                # Finalize block
                assigned_nodes = self._finalize_tendermint_block(block_proposal, job)
                self.height += 1
                self.round = 0
                
                print(f"🚀 TENDERMINT CONSENSUS FINALIZED!")
                print(f"  Job {job.name} assigned to nodes: {assigned_nodes}")
                print(f"  Block finalized at height {self.height - 1}")
                
                return ConsensusResult("Tendermint", True, job.id, assigned_nodes,
                                     time.time() - start_time, rounds_completed + 1, messages_sent, "Byzantine",
                                     {"height": self.height - 1, "rounds": rounds_completed + 1, "finalized": True})
            else:
                print(f"    ❌ Insufficient precommits: {len(precommits)}/{(2 * len(self.agents)) // 3 + 1}")
            
            self.round += 1
            rounds_completed += 1
        
        return ConsensusResult("Tendermint", False, job.id, [], time.time() - start_time,
                             rounds_completed, messages_sent, "Byzantine",
                             {"reason": "Maximum rounds exceeded", "rounds": rounds_completed})
    
    def _get_round_proposer(self) -> FaultTolerantAgent:
        """Get proposer for current round (round-robin)"""
        return self.agents[(self.height + self.round) % len(self.agents)]
    
    def _create_block_proposal(self, proposer: FaultTolerantAgent, job: HPCJob) -> Optional[Dict]:
        """Create Tendermint block proposal"""
        available_nodes = proposer.get_available_nodes()
        
        suitable_nodes = [
            node for node in available_nodes
            if (node.cpu_cores >= job.cpu_per_node and 
                node.memory_gb >= job.memory_per_node and
                node.gpu_count >= job.gpu_per_node)
        ]
        
        if len(suitable_nodes) < job.nodes_required:
            return None
        
        selected_nodes = suitable_nodes[:job.nodes_required]
        
        return {
            "height": self.height,
            "round": self.round,
            "job_id": job.id,
            "proposer": proposer.agent_id,
            "assigned_nodes": [node.id for node in selected_nodes],
            "timestamp": time.time()
        }
    
    def _compute_block_hash(self, block_proposal: Dict) -> str:
        """Compute block hash"""
        block_str = json.dumps(block_proposal, sort_keys=True)
        return hashlib.sha256(block_str.encode()).hexdigest()[:16]
    
    def _collect_prevotes(self, block_hash: str, block_proposal: Dict) -> List[str]:
        """Collect prevotes from validators"""
        prevotes = []
        
        for agent in self.agents:
            if not agent.byzantine_behavior and self._validate_block_proposal(agent, block_proposal):
                prevotes.append(agent.agent_id)
                print(f"    🗳️  {agent.agent_id} prevoted for block {block_hash[:8]}")
        
        return prevotes
    
    def _collect_precommits(self, block_hash: str, block_proposal: Dict) -> List[str]:
        """Collect precommits from validators"""
        precommits = []
        
        for agent in self.agents:
            if not agent.byzantine_behavior and self._validate_block_proposal(agent, block_proposal):
                precommits.append(agent.agent_id)
                print(f"    🔐 {agent.agent_id} precommitted block {block_hash[:8]}")
        
        return precommits
    
    def _validate_block_proposal(self, agent: FaultTolerantAgent, block_proposal: Dict) -> bool:
        """Validate block proposal"""
        # Basic validation
        return (block_proposal["height"] == self.height and
                block_proposal["round"] == self.round and
                len(block_proposal["assigned_nodes"]) > 0)
    
    def _finalize_tendermint_block(self, block_proposal: Dict, job: HPCJob) -> List[str]:
        """Finalize Tendermint block"""
        assigned_nodes = block_proposal["assigned_nodes"]
        proposer = next(a for a in self.agents if a.agent_id == block_proposal["proposer"])
        
        # Allocate nodes
        for node_id in assigned_nodes:
            node = next(n for n in proposer.managed_nodes if n.id == node_id)
            node.allocated = True
        
        job.status = "running"
        job.assigned_nodes = assigned_nodes
        
        return assigned_nodes

class AdvancedFaultTolerantSystem:
    """System to test and compare advanced fault-tolerant consensus methods"""
    
    def __init__(self):
        self.agents = []
        self.jobs = []
        self.results = []
    
    def setup_hpc_environment(self):
        """Set up realistic HPC environment"""
        print("🏗️  Setting up Advanced Fault-Tolerant HPC Environment")
        print("=" * 60)
        
        # Create 7 agents (perfect for f=2 Byzantine tolerance)
        agent_configs = [
            ("PRIMARY_CONTROLLER", 500),
            ("GPU_CLUSTER_MANAGER", 400),
            ("CPU_CLUSTER_MANAGER", 400),
            ("MEMORY_MANAGER", 300),
            ("STORAGE_MANAGER", 200),
            ("BACKUP_CONTROLLER", 150),
            ("EDGE_COORDINATOR", 100)
        ]
        
        for agent_id, stake in agent_configs:
            agent = FaultTolerantAgent(agent_id, stake)
            
            # Add nodes to each agent
            for i in range(random.randint(10, 20)):
                node = HPCNode(
                    id=f"{agent_id}_node_{i:02d}",
                    name=f"{agent_id.lower()}_node_{i:02d}",
                    cpu_cores=random.randint(16, 128),
                    memory_gb=random.randint(64, 512),
                    gpu_count=random.randint(0, 8) if "GPU" in agent_id else random.randint(0, 2),
                    storage_tb=random.randint(5, 100),
                    node_type=agent_id.split('_')[0].lower()
                )
                agent.add_node(node)
            
            self.agents.append(agent)
            total_nodes = len(agent.managed_nodes)
            total_resources = {
                "cpu": sum(n.cpu_cores for n in agent.managed_nodes),
                "memory": sum(n.memory_gb for n in agent.managed_nodes),
                "gpu": sum(n.gpu_count for n in agent.managed_nodes)
            }
            
            print(f"  👤 {agent_id}:")
            print(f"      Nodes: {total_nodes}, Stake: {stake}")
            print(f"      Resources: {total_resources['cpu']}CPU, {total_resources['memory']}GB, {total_resources['gpu']}GPU")
    
    def generate_test_jobs(self):
        """Generate diverse test jobs"""
        print(f"\n📊 Generating Test Jobs")
        
        job_templates = [
            ("AI_TRAINING", 4, 32, 256, 4, "high"),
            ("CLIMATE_SIM", 8, 16, 128, 0, "high"),  
            ("GENOMICS", 2, 8, 64, 0, "medium"),
            ("PHYSICS_SIM", 16, 8, 32, 2, "high"),
            ("DATA_ANALYTICS", 1, 16, 128, 0, "medium")
        ]
        
        for i, (name, nodes, cpu, mem, gpu, priority) in enumerate(job_templates):
            job = HPCJob(
                id=f"job_{i:03d}",
                name=f"{name}_{i}",
                nodes_required=nodes,
                cpu_per_node=cpu,
                memory_per_node=mem,
                gpu_per_node=gpu,
                runtime_hours=random.randint(1, 8),
                priority=priority,
                job_type=name.lower()
            )
            self.jobs.append(job)
        
        print(f"  📝 Generated {len(self.jobs)} test jobs")
        for job in self.jobs:
            print(f"    • {job.name}: {job.nodes_required} nodes × {job.cpu_per_node}CPU/{job.memory_per_node}GB/{job.gpu_per_node}GPU")
    
    def test_byzantine_faults(self):
        """Test system behavior with Byzantine faults"""
        print(f"\n💀 Testing Byzantine Fault Tolerance")
        print("=" * 50)
        
        # Introduce Byzantine behavior in 1-2 agents  
        byzantine_count = min(2, len(self.agents) // 3)
        byzantine_agents = random.sample(self.agents, byzantine_count)
        
        for agent in byzantine_agents:
            agent.byzantine_behavior = True
            print(f"  ☠️  {agent.agent_id} is now Byzantine (malicious)")
        
        print(f"  🛡️  System should tolerate {len(self.agents) // 3} Byzantine failures")
    
    def run_comprehensive_comparison(self):
        """Run all three advanced consensus methods"""
        print(f"\n🔬 COMPREHENSIVE FAULT-TOLERANT CONSENSUS COMPARISON")
        print("=" * 80)
        
        protocols = [
            ("PBFT", lambda job: PBFTConsensus(self.agents).run_consensus(job)),
            ("Multi-Paxos", lambda job: MultiPaxosConsensus(self.agents).run_consensus(job)),
            ("Tendermint", lambda job: TendermintConsensus(self.agents).run_consensus(job))
        ]
        
        all_results = {}
        
        for protocol_name, protocol_func in protocols:
            print(f"\n{'='*70}")
            print(f"🧪 TESTING {protocol_name.upper()} PROTOCOL")
            print(f"{'='*70}")
            
            # Reset node allocations
            for agent in self.agents:
                for node in agent.managed_nodes:
                    node.allocated = False
            
            protocol_results = []
            
            # Test each job
            for job in self.jobs:
                print(f"\n--- Job: {job.name} ({job.nodes_required} nodes required) ---")
                
                try:
                    result = protocol_func(job)
                    protocol_results.append(result)
                    
                    if result.success:
                        print(f"  ✅ SUCCESS: {len(result.assigned_nodes)} nodes allocated")
                        print(f"      Time: {result.time_taken:.3f}s, Messages: {result.messages_sent}, Rounds: {result.rounds}")
                    else:
                        print(f"  ❌ FAILED: {result.details.get('reason', 'Unknown reason')}")
                        
                except Exception as e:
                    print(f"  💥 ERROR: {str(e)}")
                    result = ConsensusResult(protocol_name, False, job.id, [], 0, 0, 0, "", {"error": str(e)})
                    protocol_results.append(result)
                
                time.sleep(0.1)  # Brief pause
            
            all_results[protocol_name] = protocol_results
        
        # Comprehensive analysis
        self._analyze_fault_tolerant_results(all_results)
    
    def _analyze_fault_tolerant_results(self, results: Dict[str, List[ConsensusResult]]):
        """Analyze and compare fault-tolerant consensus results"""
        print(f"\n{'='*80}")
        print("📈 ADVANCED FAULT-TOLERANT CONSENSUS ANALYSIS")
        print(f"{'='*80}")
        
        protocol_stats = {}
        
        for protocol_name, protocol_results in results.items():
            successful = sum(1 for r in protocol_results if r.success)
            total_time = sum(r.time_taken for r in protocol_results)
            total_messages = sum(r.messages_sent for r in protocol_results)
            total_rounds = sum(r.rounds for r in protocol_results)
            
            avg_time = total_time / len(protocol_results) if protocol_results else 0
            avg_messages = total_messages / len(protocol_results) if protocol_results else 0
            avg_rounds = total_rounds / len(protocol_results) if protocol_results else 0
            
            protocol_stats[protocol_name] = {
                'success_rate': (successful / len(protocol_results)) * 100 if protocol_results else 0,
                'avg_time': avg_time,
                'avg_messages': avg_messages,
                'avg_rounds': avg_rounds,
                'total_jobs': len(protocol_results),
                'successful_jobs': successful,
                'fault_tolerance': protocol_results[0].fault_tolerance if protocol_results else "Unknown"
            }
        
        # Print detailed results
        for protocol_name, stats in protocol_stats.items():
            print(f"\n🏆 {protocol_name.upper()} RESULTS:")
            print(f"  ✅ Success Rate: {stats['success_rate']:.1f}%")
            print(f"  ⏱️  Average Time: {stats['avg_time']:.3f}s")
            print(f"  📨 Average Messages: {stats['avg_messages']:.1f}")
            print(f"  🔄 Average Rounds: {stats['avg_rounds']:.1f}")
            print(f"  🛡️  Fault Model: {stats['fault_tolerance']}")
            print(f"  📊 Jobs: {stats['successful_jobs']}/{stats['total_jobs']}")
        
        # Rankings
        print(f"\n🥇 PROTOCOL RANKINGS:")
        
        # By success rate
        by_success = sorted(protocol_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        print(f"\n  📈 By Success Rate:")
        for rank, (protocol, stats) in enumerate(by_success, 1):
            print(f"    #{rank} {protocol}: {stats['success_rate']:.1f}%")
        
        # By speed
        by_speed = sorted(protocol_stats.items(), key=lambda x: x[1]['avg_time'])
        print(f"\n  🚀 By Speed:")
        for rank, (protocol, stats) in enumerate(by_speed, 1):
            print(f"    #{rank} {protocol}: {stats['avg_time']:.3f}s")
        
        # By efficiency (messages)
        by_efficiency = sorted(protocol_stats.items(), key=lambda x: x[1]['avg_messages'])
        print(f"\n  📡 By Message Efficiency:")
        for rank, (protocol, stats) in enumerate(by_efficiency, 1):
            print(f"    #{rank} {protocol}: {stats['avg_messages']:.1f} messages")
        
        # Recommendations
        print(f"\n💡 FAULT-TOLERANT CONSENSUS RECOMMENDATIONS:")
        print(f"  🔒 For Maximum Byzantine Tolerance: PBFT or Tendermint")
        print(f"  🏛️  For Proven Crash Tolerance: Multi-Paxos")
        print(f"  ⚡ For Speed + Byzantine Tolerance: Tendermint")
        print(f"  📡 For Message Efficiency: Multi-Paxos")
        print(f"  🎯 For High Availability: PBFT")
        
        # Fault tolerance comparison
        print(f"\n🛡️  FAULT TOLERANCE COMPARISON:")
        print(f"  PBFT: Tolerates ⌊(n-1)/3⌋ Byzantine failures")
        print(f"  Multi-Paxos: Tolerates ⌊(n-1)/2⌋ crash failures")  
        print(f"  Tendermint: Tolerates ⌊(n-1)/3⌋ Byzantine failures + immediate finality")
        
        print(f"\n🎯 For HPC Systems:")
        print(f"  • Use PBFT for security-critical multi-tenant environments")
        print(f"  • Use Multi-Paxos for reliable job queue management")
        print(f"  • Use Tendermint for real-time resource allocation with finality")

def main():
    """Main execution function"""
    print("🌟 Advanced Fault-Tolerant Multi-Agent HPC Consensus Demo")
    print("="*80)
    
    # Initialize system
    system = AdvancedFaultTolerantSystem()
    
    # Setup environment
    system.setup_hpc_environment()
    
    # Generate workloads
    system.generate_test_jobs()
    
    # Introduce Byzantine faults
    system.test_byzantine_faults()
    
    # Run comprehensive comparison
    system.run_comprehensive_comparison()
    
    print(f"\n🏁 Advanced fault-tolerant consensus demonstration complete!")
    print("="*80)

if __name__ == "__main__":
    main()

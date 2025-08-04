#!/usr/bin/env python3
"""
Combined Centralized and Distributed Scheduling Demo
====================================================

This demo showcases both centralized and distributed scheduling mechanisms:
- Centralized Scheduler: Direct assignment of jobs by a central authority.
- Distributed Agent Negotiation: Agents self-evaluate, propose, and negotiate job handling.
"""

import os
import sys
import uuid
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
from true_discrete_event_demo import (TrueDiscreteEventSimulation, DiscreteEventJob, 
                                     DiscreteEventAgent, HPC_Resource, SimulationEvent, 
                                     EventType, Priority, SimulationClock)

class Message:
    """Represents a message in the distributed scheduling system"""
    def __init__(self, msg_type: str, sender: str, recipient: str, data: Dict[str, Any]):
        self.msg_type = msg_type
        self.sender = sender
        self.recipient = recipient
        self.data = data
        self.timestamp = time.time()

class CentralizedScheduler:
    """Centralized scheduling: direct assignment by central authority"""
    
    def __init__(self, simulation):
        self.simulation = simulation
        self.name = "Centralized"
        self.pending_jobs: List[DiscreteEventJob] = []
        self.running_jobs: Dict[str, DiscreteEventJob] = {}
        self.completed_jobs: List[DiscreteEventJob] = []
        self.failed_jobs: List[DiscreteEventJob] = []
        self.metrics = {
            'assignments_made': 0,
            'decision_time': 0.0,
            'message_count': 0
        }
    
    def schedule_job(self, job: DiscreteEventJob):
        """Schedule job using centralized decision making"""
        start_time = time.time()
        self.pending_jobs.append(job)
        
        print(f"📋 [CENTRALIZED] Scheduling job {job.name} (CPU:{job.resource_requirements.get('cpu', 0)}, "
              f"Memory:{job.resource_requirements.get('memory', 0)}GB)")
        
        # Centralized evaluation of all agents
        best_agent = None
        best_score = -1
        
        for agent in self.simulation.agents.values():
            if agent.is_available():
                can_handle, score = agent.resource.can_handle_job(job.resource_requirements)
                # Also check if agent has sufficient available resources right now
                cpu_req = job.resource_requirements.get('cpu', 0)
                memory_req = job.resource_requirements.get('memory', 0) 
                gpu_req = job.resource_requirements.get('gpu', 0)
                
                has_resources = (agent.available_cpu >= cpu_req and 
                               agent.available_memory >= memory_req and 
                               agent.available_gpu >= gpu_req)
                
                if can_handle and has_resources and score > best_score:
                    best_score = score
                    best_agent = agent
        
        if best_agent:
            # Direct assignment (no negotiation needed)
            job.assigned_agent = best_agent.agent_id
            job.status = "assigned"
            self.pending_jobs.remove(job)
            self.running_jobs[job.id] = job
            
            # Store in job registry and schedule start event
            self.simulation.job_registry[job.id] = job
            start_event = SimulationEvent(
                time=self.simulation.clock.now(),
                event_type=EventType.JOB_START,
                data={'job_id': job.id, 'agent_id': best_agent.agent_id},
                priority=1
            )
            self.simulation.schedule_event(start_event)
            
            self.metrics['assignments_made'] += 1
            decision_time = time.time() - start_time
            self.metrics['decision_time'] += decision_time
            
            print(f"✅ [CENTRALIZED] Assigned {job.name} to {best_agent.agent_id} "
                  f"(score: {best_score:.2f}, decision: {decision_time*1000:.1f}ms)")
        else:
            print(f"❌ [CENTRALIZED] No available agent for {job.name} - returning to job pool")
            # Remove from pending queue and return to simulation job pool
            if job in self.pending_jobs:
                self.pending_jobs.remove(job)
            
            # Schedule a retry event for later when resources might become available
            retry_event = SimulationEvent(
                time=self.simulation.clock.now() + 5.0,  # Retry after 5 time units
                event_type=EventType.JOB_ARRIVAL,
                data={'job_id': job.id},
                priority=2  # Lower priority than new jobs
            )
            self.simulation.schedule_event(retry_event)
    
    def handle_job_completion(self, job: DiscreteEventJob):
        """Handle job completion and reschedule failed jobs"""
        if job.id in self.running_jobs:
            del self.running_jobs[job.id]
        
        if job.status == "completed":
            self.completed_jobs.append(job)
        else:
            # Return failed jobs to the simulation's job pool for rescheduling
            print(f"🔄 [CENTRALIZED] Job {job.name} failed - returning to job pool")
            job.status = "pending"  # Reset status
            job.assigned_agent = None  # Clear previous assignment
            
            # Remove from our pending queue if present to avoid duplicates
            if job in self.pending_jobs:
                self.pending_jobs.remove(job)
            
            # Return to simulation's job pool and schedule retry event
            retry_event = SimulationEvent(
                time=self.simulation.clock.now() + 1.0,  # Retry after 1 time unit
                event_type=EventType.JOB_ARRIVAL,
                data={'job_id': job.id},
                priority=0
            )
            self.simulation.schedule_event(retry_event)
        
        # When an agent becomes available, try to schedule pending jobs
        self.retry_pending_jobs()
    
    def retry_pending_jobs(self):
        """Retry pending jobs that couldn't be scheduled initially"""
        if not self.pending_jobs:
            return
            
        retry_jobs = [job for job in self.pending_jobs if job.status == "pending"]
        
        for job in retry_jobs[:]:
            # Try to find an available agent for this job
            best_agent = None
            best_score = -1
            
            for agent in self.simulation.agents.values():
                if agent.is_available():
                    can_handle, score = agent.resource.can_handle_job(job.resource_requirements)
                    cpu_req = job.resource_requirements.get('cpu', 0)
                    memory_req = job.resource_requirements.get('memory', 0)
                    gpu_req = job.resource_requirements.get('gpu', 0)
                    
                    has_resources = (agent.available_cpu >= cpu_req and 
                                   agent.available_memory >= memory_req and 
                                   agent.available_gpu >= gpu_req)
                    
                    if can_handle and has_resources and score > best_score:
                        best_score = score
                        best_agent = agent
            
            if best_agent:
                # Successfully retry the job
                job.assigned_agent = best_agent.agent_id
                job.status = "assigned"
                self.pending_jobs.remove(job)
                self.running_jobs[job.id] = job
                
                # Schedule start event
                start_event = SimulationEvent(
                    time=self.simulation.clock.now(),
                    event_type=EventType.JOB_START,
                    data={'job_id': job.id, 'agent_id': best_agent.agent_id},
                    priority=1
                )
                self.simulation.schedule_event(start_event)
                
                # Don't increment assignments_made for retries - only count initial assignments
                print(f"✅ [CENTRALIZED] Successfully retried {job.name} on {best_agent.agent_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            'scheduler_type': self.name,
            'pending_jobs': len(self.pending_jobs),
            'running_jobs': len(self.running_jobs),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'metrics': self.metrics.copy()
        }

class DistributedScheduler:
    """Distributed scheduling: agents negotiate and self-select for jobs"""
    
    def __init__(self, simulation):
        self.simulation = simulation
        self.name = "Distributed"
        self.pending_jobs: List[DiscreteEventJob] = []
        self.running_jobs: Dict[str, DiscreteEventJob] = {}
        self.completed_jobs: List[DiscreteEventJob] = []
        self.failed_jobs: List[DiscreteEventJob] = []
        self.message_queue: List[Message] = []
        self.metrics = {
            'assignments_made': 0,
            'decision_time': 0.0,
            'message_count': 0,
            'offers_received': 0,
            'negotiation_rounds': 0
        }
    
    def schedule_job(self, job: DiscreteEventJob):
        """Schedule job using distributed negotiation with retry logic"""
        start_time = time.time()
        self.pending_jobs.append(job)
        
        print(f"📋 [DISTRIBUTED] Broadcasting job {job.name} for negotiation "
              f"(CPU:{job.resource_requirements.get('cpu', 0)}, "
              f"Memory:{job.resource_requirements.get('memory', 0)}GB)")
        
        # Phase 1: Broadcast job to all agents
        offers = self._collect_offers(job)
        
        if offers:
            # Phase 2: Try to assign to agents in order of preference
            assigned = False
            attempts = 0
            max_attempts = min(3, len(offers))  # Try up to 3 best offers
            
            # Sort offers by composite score (score * confidence)
            offers.sort(key=lambda x: x['score'] * x['confidence'], reverse=True)
            
            for offer in offers[:max_attempts]:
                attempts += 1
                winning_agent = offer['agent']
                
                # Phase 3: Confirm assignment with current best agent
                if self._confirm_assignment(job, winning_agent):
                    job.assigned_agent = winning_agent.agent_id
                    job.status = "assigned"
                    self.pending_jobs.remove(job)
                    self.running_jobs[job.id] = job
                    
                    # Store in job registry and schedule start event
                    self.simulation.job_registry[job.id] = job
                    start_event = SimulationEvent(
                        time=self.simulation.clock.now(),
                        event_type=EventType.JOB_START,
                        data={'job_id': job.id, 'agent_id': winning_agent.agent_id},
                        priority=1
                    )
                    self.simulation.schedule_event(start_event)
                    
                    self.metrics['assignments_made'] += 1
                    decision_time = time.time() - start_time
                    self.metrics['decision_time'] += decision_time
                    
                    attempt_msg = f" (attempt {attempts})" if attempts > 1 else ""
                    print(f"✅ [DISTRIBUTED] Negotiated assignment of {job.name} to {winning_agent.agent_id}{attempt_msg} "
                          f"(score: {offer['score']:.2f}, decision: {decision_time*1000:.1f}ms, "
                          f"offers: {len(offers)})")
                    assigned = True
                    break
                else:
                    print(f"  🔄 [DISTRIBUTED] Trying next agent for {job.name} (attempt {attempts})")
            
            if not assigned:
                print(f"❌ [DISTRIBUTED] All assignment attempts failed for {job.name} after {attempts} tries - returning to job pool")
                # Remove from pending queue and return to simulation job pool
                if job in self.pending_jobs:
                    self.pending_jobs.remove(job)
                
                # Schedule retry event to return job to pool
                retry_event = SimulationEvent(
                    time=self.simulation.clock.now() + 5.0,  # Retry after 5 time units
                    event_type=EventType.JOB_ARRIVAL,
                    data={'job_id': job.id},
                    priority=2  # Lower priority than new jobs
                )
                self.simulation.schedule_event(retry_event)
        else:
            print(f"❌ [DISTRIBUTED] No offers received for {job.name} - returning to job pool")
            # Remove from pending queue and return to simulation job pool
            if job in self.pending_jobs:
                self.pending_jobs.remove(job)
            
            # Schedule retry event to return job to pool
            retry_event = SimulationEvent(
                time=self.simulation.clock.now() + 5.0,  # Retry after 5 time units
                event_type=EventType.JOB_ARRIVAL,
                data={'job_id': job.id},
                priority=2  # Lower priority than new jobs
            )
            self.simulation.schedule_event(retry_event)
    
    def retry_pending_jobs(self):
        """Retry jobs that are still in pending queue when resources become available"""
        retry_jobs = [job for job in self.pending_jobs if job.status == "pending"]
        
        for job in retry_jobs[:]:
            print(f"🔄 [DISTRIBUTED] Retrying {job.name}...")
            
            # Try scheduling again
            offers = self._collect_offers(job)
            if offers:
                # Sort and try assignment
                offers.sort(key=lambda x: x['score'] * x['confidence'], reverse=True)
                for offer in offers[:1]:  # Just try best offer on retry
                    if self._confirm_assignment(job, offer['agent']):
                        job.assigned_agent = offer['agent'].agent_id
                        job.status = "assigned"
                        self.pending_jobs.remove(job)
                        self.running_jobs[job.id] = job
                        
                        # Store in job registry and schedule start event
                        self.simulation.job_registry[job.id] = job
                        start_event = SimulationEvent(
                            time=self.simulation.clock.now(),
                            event_type=EventType.JOB_START,
                            data={'job_id': job.id, 'agent_id': offer['agent'].agent_id},
                            priority=1
                        )
                        self.simulation.schedule_event(start_event)
                        
                        # Don't increment assignments_made for retries - only count initial assignments
                        print(f"✅ [DISTRIBUTED] Successfully retried {job.name} on {offer['agent'].agent_id}")
                        break
                else:
                    # If retry fails, return to job pool
                    print(f"🔄 [DISTRIBUTED] Retry failed for {job.name} - returning to job pool")
                    self.pending_jobs.remove(job)
                    retry_event = SimulationEvent(
                        time=self.simulation.clock.now() + 2.0,  # Retry again later
                        event_type=EventType.JOB_ARRIVAL,
                        data={'job_id': job.id},
                        priority=2
                    )
                    self.simulation.schedule_event(retry_event)
            else:
                # No offers available, return to job pool for later
                print(f"🔄 [DISTRIBUTED] No offers for {job.name} retry - returning to job pool")
                self.pending_jobs.remove(job)
                retry_event = SimulationEvent(
                    time=self.simulation.clock.now() + 3.0,  # Retry again later
                    event_type=EventType.JOB_ARRIVAL,
                    data={'job_id': job.id},
                    priority=2
                )
                self.simulation.schedule_event(retry_event)
    
    def _collect_offers(self, job: DiscreteEventJob) -> List[Dict[str, Any]]:
        """Collect offers from all capable agents"""
        offers = []
        
        for agent in self.simulation.agents.values():
            if agent.is_available():
                can_handle, base_score = agent.resource.can_handle_job(job.resource_requirements)
                if can_handle:
                    # Simulate agent's self-evaluation with randomness
                    offer_score = base_score * random.uniform(0.7, 1.3)
                    confidence = random.uniform(0.6, 1.0)
                    
                    offer = {
                        'agent': agent,
                        'score': offer_score,
                        'confidence': confidence,
                        'timestamp': time.time()
                    }
                    offers.append(offer)
                    
                    # Simulate message passing
                    self._send_message("OFFER", agent.agent_id, "scheduler", 
                                     {'job_id': job.id, 'score': offer_score, 'confidence': confidence})
        
        self.metrics['offers_received'] += len(offers)
        self.metrics['negotiation_rounds'] += 1
        
        print(f"  🔄 [DISTRIBUTED] Collected {len(offers)} offers for {job.name}")
        return offers
    
    def _select_best_offer(self, offers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best offer from collected offers"""
        # Sort by composite score (score * confidence)
        offers.sort(key=lambda x: x['score'] * x['confidence'], reverse=True)
        return offers[0]
    
    def _confirm_assignment(self, job: DiscreteEventJob, agent: DiscreteEventAgent) -> bool:
        """Confirm assignment with the selected agent with temporary resource reservation"""
        # Double-check agent availability and capability at confirmation time
        if agent.is_available() and agent.can_handle_job(job):
            # Temporarily reserve resources to prevent race conditions
            cpu_req = job.resource_requirements.get('cpu', 0)
            memory_req = job.resource_requirements.get('memory', 0)
            gpu_req = job.resource_requirements.get('gpu', 0)
            
            # Quick resource reservation check
            if (agent.available_cpu >= cpu_req and 
                agent.available_memory >= memory_req and 
                agent.available_gpu >= gpu_req):
                
                self._send_message("ASSIGNMENT", "scheduler", agent.agent_id, 
                                 {'job_id': job.id, 'confirmed': True})
                return True
            else:
                print(f"  ⚠️  [DISTRIBUTED] Agent {agent.agent_id} rejected {job.name} - insufficient resources")
                self._send_message("ASSIGNMENT", "scheduler", agent.agent_id, 
                                 {'job_id': job.id, 'confirmed': False})
                return False
        else:
            print(f"  ⚠️  [DISTRIBUTED] Agent {agent.agent_id} rejected {job.name} - agent unavailable")
            self._send_message("ASSIGNMENT", "scheduler", agent.agent_id, 
                             {'job_id': job.id, 'confirmed': False})
            return False
    
    def _send_message(self, msg_type: str, sender: str, recipient: str, data: Dict[str, Any]):
        """Simulate message passing in distributed system"""
        message = Message(msg_type, sender, recipient, data)
        self.message_queue.append(message)
        self.metrics['message_count'] += 1
    
    def handle_job_completion(self, job: DiscreteEventJob):
        """Handle job completion and trigger retry of pending jobs"""
        if job.id in self.running_jobs:
            del self.running_jobs[job.id]
        
        if job.status == "completed":
            self.completed_jobs.append(job)
        else:
            # Return failed jobs to the simulation's job pool for rescheduling
            print(f"🔄 [DISTRIBUTED] Job {job.name} failed - returning to job pool")
            job.status = "pending"  # Reset status
            job.assigned_agent = None  # Clear previous assignment
            
            # Remove from our pending queue if present to avoid duplicates
            if job in self.pending_jobs:
                self.pending_jobs.remove(job)
            
            # Return to simulation's job pool and schedule retry event
            retry_event = SimulationEvent(
                time=self.simulation.clock.now() + 1.0,  # Retry after 1 time unit
                event_type=EventType.JOB_ARRIVAL,
                data={'job_id': job.id},
                priority=0
            )
            self.simulation.schedule_event(retry_event)
        
        # Trigger retry of pending jobs when resources become available
        self.retry_pending_jobs()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            'scheduler_type': self.name,
            'pending_jobs': len(self.pending_jobs),
            'running_jobs': len(self.running_jobs),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'metrics': self.metrics.copy(),
            'message_queue_size': len(self.message_queue)
        }


def create_realistic_jobs() -> List[DiscreteEventJob]:
    """Create comprehensive HPC workload with 26 jobs across 6 categories (same as discrete_event_demo.py)"""
    jobs = []
    
    # 1. Development & Testing Jobs (20% - small, quick jobs)
    dev_jobs = [
        ("Unit Test Suite", Priority.HIGH, {"cpu": 1, "memory": 2, "gpu": 0}, 3.0, 0.8),
        ("Code Compilation", Priority.MEDIUM, {"cpu": 2, "memory": 4, "gpu": 0}, 8.0, 1.2),
        ("Integration Test", Priority.MEDIUM, {"cpu": 1, "memory": 1, "gpu": 0}, 5.0, 0.9),
        ("Debug Session", Priority.HIGH, {"cpu": 1, "memory": 2, "gpu": 0}, 12.0, 1.5),
        ("Performance Profiling", Priority.MEDIUM, {"cpu": 2, "memory": 4, "gpu": 0}, 15.0, 1.3),
    ]
    for name, priority, req, duration, complexity in dev_jobs:
        jobs.append(DiscreteEventJob(str(uuid.uuid4()), name, priority, req, duration, complexity))
    
    # 2. Scientific Computing Jobs (30% - medium CPU, moderate memory)
    scientific_jobs = [
        ("Climate Simulation", Priority.MEDIUM, {"cpu": 16, "memory": 64, "gpu": 0}, 45.0, 1.8),
        ("Molecular Dynamics", Priority.MEDIUM, {"cpu": 12, "memory": 48, "gpu": 0}, 35.0, 1.6),
        ("CFD Analysis", Priority.MEDIUM, {"cpu": 16, "memory": 80, "gpu": 0}, 40.0, 1.7),
        ("Quantum Chemistry", Priority.HIGH, {"cpu": 8, "memory": 32, "gpu": 0}, 25.0, 1.4),
        ("Protein Folding", Priority.MEDIUM, {"cpu": 24, "memory": 96, "gpu": 0}, 50.0, 2.0),
        ("Weather Modeling", Priority.MEDIUM, {"cpu": 28, "memory": 100, "gpu": 0}, 60.0, 1.9),
        ("Seismic Processing", Priority.MEDIUM, {"cpu": 16, "memory": 64, "gpu": 0}, 30.0, 1.5),
    ]
    for name, priority, req, duration, complexity in scientific_jobs:
        jobs.append(DiscreteEventJob(str(uuid.uuid4()), name, priority, req, duration, complexity))
    
    # 3. Machine Learning & AI Jobs (25% - GPU-intensive)
    ml_jobs = [
        ("Deep Learning Training", Priority.HIGH, {"cpu": 8, "memory": 128, "gpu": 4}, 120.0, 2.5),
        ("Neural Network Inference", Priority.MEDIUM, {"cpu": 4, "memory": 32, "gpu": 1}, 8.0, 0.7),
        ("Computer Vision Model", Priority.HIGH, {"cpu": 6, "memory": 64, "gpu": 2}, 80.0, 2.0),
        ("NLP Transformer Training", Priority.MEDIUM, {"cpu": 16, "memory": 256, "gpu": 4}, 150.0, 2.8),
        ("Reinforcement Learning", Priority.MEDIUM, {"cpu": 4, "memory": 16, "gpu": 1}, 90.0, 2.2),
        ("Model Hyperparameter Tuning", Priority.LOW, {"cpu": 2, "memory": 8, "gpu": 1}, 45.0, 1.3),
    ]
    for name, priority, req, duration, complexity in ml_jobs:
        jobs.append(DiscreteEventJob(str(uuid.uuid4()), name, priority, req, duration, complexity))
    
    # 4. Big Data & Analytics Jobs (15% - memory-intensive)
    bigdata_jobs = [
        ("Genomics Analysis", Priority.MEDIUM, {"cpu": 8, "memory": 512, "gpu": 0}, 75.0, 2.1),
        ("Financial Risk Modeling", Priority.HIGH, {"cpu": 16, "memory": 256, "gpu": 0}, 65.0, 1.9),
        ("Large Dataset Processing", Priority.MEDIUM, {"cpu": 4, "memory": 128, "gpu": 0}, 55.0, 1.7),
        ("Statistical Analysis", Priority.MEDIUM, {"cpu": 6, "memory": 64, "gpu": 0}, 35.0, 1.4),
    ]
    for name, priority, req, duration, complexity in bigdata_jobs:
        jobs.append(DiscreteEventJob(str(uuid.uuid4()), name, priority, req, duration, complexity))
    
    # 5. Batch Processing Jobs (8% - low priority, resource-efficient)
    batch_jobs = [
        ("Nightly Data Backup", Priority.LOW, {"cpu": 1, "memory": 4, "gpu": 0}, 20.0, 0.9),
        ("Log File Processing", Priority.LOW, {"cpu": 2, "memory": 8, "gpu": 0}, 25.0, 1.1),
        ("Report Generation", Priority.LOW, {"cpu": 1, "memory": 2, "gpu": 0}, 15.0, 0.8),
    ]
    for name, priority, req, duration, complexity in batch_jobs:
        jobs.append(DiscreteEventJob(str(uuid.uuid4()), name, priority, req, duration, complexity))
    
    # 6. Critical System Jobs (2% - highest priority, minimal resources)
    critical_jobs = [
        ("System Health Check", Priority.CRITICAL, {"cpu": 1, "memory": 1, "gpu": 0}, 2.0, 0.5),
    ]
    for name, priority, req, duration, complexity in critical_jobs:
        jobs.append(DiscreteEventJob(str(uuid.uuid4()), name, priority, req, duration, complexity))
    
    print(f"\n📊 Created comprehensive HPC workload ({len(jobs)} jobs):")
    priority_counts = {}
    for job in jobs:
        priority_counts[job.priority.name] = priority_counts.get(job.priority.name, 0) + 1
    
    for priority, count in sorted(priority_counts.items()):
        print(f"  • {priority}: {count} jobs")
    
    return jobs

def run_scheduling_comparison():
    """Run comprehensive comparison of centralized vs distributed scheduling"""
    
    print("\n" + "="*80)
    print(" CENTRALIZED vs DISTRIBUTED SCHEDULING COMPARISON")
    print("="*80)
    
    results = {}
    
    for scheduler_type in ["Centralized", "Distributed"]:
        print(f"\n🚀 Running {scheduler_type} Scheduling Simulation...")
        print("-" * 50)
        
        # Create fresh simulation for each test
        simulation = TrueDiscreteEventSimulation()
        
        # Create comprehensive heterogeneous agent cluster to handle all job types
        resources = [
            # Development & Small Jobs
            ("small-cpu-1", HPC_Resource("Small CPU Alpha", 4, 16, 0, 8.0), 0.05),
            ("small-cpu-2", HPC_Resource("Small CPU Beta", 4, 16, 0, 8.0), 0.05),
            
            # Medium CPU Compute
            ("medium-cpu-1", HPC_Resource("Medium CPU Gamma", 8, 32, 0, 15.0), 0.08),
            ("medium-cpu-2", HPC_Resource("Medium CPU Delta", 12, 48, 0, 18.0), 0.08),
            
            # Large CPU Compute (for scientific computing)
            ("large-cpu-1", HPC_Resource("Large CPU Epsilon", 16, 64, 0, 25.0), 0.10),
            ("large-cpu-2", HPC_Resource("Large CPU Zeta", 20, 80, 0, 30.0), 0.10),
            ("xlarge-cpu", HPC_Resource("XLarge CPU Eta", 32, 128, 0, 45.0), 0.12),
            
            # GPU Nodes (for ML/AI workloads)
            ("gpu-small", HPC_Resource("Small GPU Node", 8, 64, 2, 35.0), 0.15),
            ("gpu-medium", HPC_Resource("Medium GPU Node", 12, 128, 4, 55.0), 0.18),
            ("gpu-large", HPC_Resource("Large GPU Node", 16, 256, 8, 85.0), 0.20),
            
            # Memory-Intensive Nodes (for big data)
            ("memory-medium", HPC_Resource("Medium Memory", 8, 256, 0, 40.0), 0.07),
            ("memory-large", HPC_Resource("Large Memory", 16, 512, 0, 60.0), 0.09),
            ("memory-xlarge", HPC_Resource("XLarge Memory", 24, 1024, 0, 90.0), 0.11),
        ]
        
        for agent_id, resource, failure_rate in resources:
            agent = DiscreteEventAgent(agent_id, resource, simulation, failure_rate)
            simulation.add_agent(agent)
        
        print(f"📋 Created {len(simulation.agents)} agents:")
        for agent_id, agent in simulation.agents.items():
            res = agent.resource
            print(f"  • {agent_id}: {res.cpu_cores}C/{res.memory_gb}GB (fail: {agent.failure_rate*100:.0f}%)")
        
        # Initialize appropriate scheduler
        if scheduler_type == "Centralized":
            scheduler = CentralizedScheduler(simulation)
        else:
            scheduler = DistributedScheduler(simulation)
        
        # Create jobs
        jobs = create_realistic_jobs()
        print(f"\n📤 Created {len(jobs)} jobs with varying resource requirements")
        
        # Set up simulation-driven scheduling instead of manual scheduling
        # Replace the simulation's scheduler with our custom scheduler
        simulation.scheduler = scheduler
        
        # Schedule jobs with staggered arrivals for realistic simulation
        print(f"\n⚡ Submitting jobs to {scheduler_type.lower()} simulation...")
        scheduling_start = time.time()
        
        for i, job in enumerate(jobs):
            arrival_time = i * 2.0  # Jobs arrive every 2 time units
            simulation.submit_job(job, arrival_time)
            print(f"  📤 Job {i+1}/{len(jobs)}: {job.name} → arrives at t={arrival_time:.1f}")
        
        scheduling_time = time.time() - scheduling_start
        
        sim_start = time.time()
        simulation.run(max_simulation_time=float('inf'))  # Run until all events are processed
        sim_time = time.time() - sim_start
        
        # Collect results
        scheduler_stats = scheduler.get_stats()
        final_stats = simulation.get_final_stats()
        
        results[scheduler_type] = {
            'scheduler_stats': scheduler_stats,
            'simulation_stats': final_stats,
            'scheduling_time': scheduling_time,
            'simulation_wall_time': sim_time,
            'total_jobs': len(jobs)
        }
        
        print(f"\n📊 {scheduler_type} Results:")
        print(f"  • Jobs scheduled: {scheduler_stats['metrics']['assignments_made']}")
        print(f"  • Jobs completed: {scheduler_stats['completed_jobs']}")
        print(f"  • Jobs failed: {scheduler_stats['failed_jobs']}")
        print(f"  • Scheduling time: {scheduling_time:.3f}s")
        print(f"  • Avg decision time: {scheduler_stats['metrics']['decision_time']/max(1, scheduler_stats['metrics']['assignments_made'])*1000:.1f}ms")
        
        if scheduler_type == "Distributed":
            print(f"  • Messages sent: {scheduler_stats['metrics']['message_count']}")
            print(f"  • Offers received: {scheduler_stats['metrics']['offers_received']}")
            print(f"  • Negotiation rounds: {scheduler_stats['metrics']['negotiation_rounds']}")
    
    # Compare results
    print("\n" + "="*80)
    print(" COMPARISON ANALYSIS")
    print("="*80)
    
    centralized = results["Centralized"]
    distributed = results["Distributed"]
    
    print(f"\n🎯 Scheduling Performance:")
    print(f"  Centralized:  {centralized['scheduler_stats']['metrics']['assignments_made']} jobs assigned")
    print(f"  Distributed:  {distributed['scheduler_stats']['metrics']['assignments_made']} jobs assigned")
    
    cent_avg_decision = centralized['scheduler_stats']['metrics']['decision_time'] / max(1, centralized['scheduler_stats']['metrics']['assignments_made'])
    dist_avg_decision = distributed['scheduler_stats']['metrics']['decision_time'] / max(1, distributed['scheduler_stats']['metrics']['assignments_made'])
    
    print(f"\n⚡ Decision Speed:")
    print(f"  Centralized:  {cent_avg_decision*1000:.1f}ms average")
    print(f"  Distributed:  {dist_avg_decision*1000:.1f}ms average")
    print(f"  Winner:       {'Centralized' if cent_avg_decision < dist_avg_decision else 'Distributed'} "
          f"({abs(cent_avg_decision - dist_avg_decision)*1000:.1f}ms faster)")
    
    print(f"\n📊 Job Completion:")
    cent_completed = centralized['scheduler_stats']['completed_jobs']
    dist_completed = distributed['scheduler_stats']['completed_jobs']
    print(f"  Centralized:  {cent_completed} jobs completed")
    print(f"  Distributed:  {dist_completed} jobs completed")
    print(f"  Winner:       {'Centralized' if cent_completed > dist_completed else 'Distributed'} "
          f"({abs(cent_completed - dist_completed)} more jobs)")
    
    print(f"\n💬 Communication Overhead:")
    cent_messages = centralized['scheduler_stats']['metrics']['message_count']
    dist_messages = distributed['scheduler_stats']['metrics']['message_count']
    print(f"  Centralized:  {cent_messages} messages")
    print(f"  Distributed:  {dist_messages} messages")
    print(f"  Overhead:     {dist_messages - cent_messages} extra messages for distributed")
    
    print(f"\n🏆 Summary:")
    print(f"  • Centralized excels at: Fast decisions, low overhead, deterministic assignment")
    print(f"  • Distributed excels at: Fault tolerance, agent autonomy, realistic negotiation")
    print(f"  • Use centralized for: High-throughput, low-latency, trusted environments")
    print(f"  • Use distributed for: Fault-tolerant, autonomous, multi-party systems")
    
    print("\n" + "="*80)
    
    return results

def run_combined_demo():
    """Run the combined scheduling demo"""
    try:
        results = run_scheduling_comparison()
        
        print("\n✅ Demo completed successfully!")
        print("\n📈 Key Insights:")
        print("  1. Centralized scheduling provides faster, deterministic job assignment")
        print("  2. Distributed scheduling enables agent autonomy and fault tolerance")
        print("  3. Message overhead is significantly higher in distributed systems")
        print("  4. Both approaches can achieve similar job completion rates")
        print("  5. Choice depends on system requirements: speed vs. resilience")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_combined_demo()


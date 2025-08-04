#!/usr/bin/env python3
"""
True Discrete Event Scheduling Demo
===================================

This demo showcases a TRUE discrete event simulation system that:
- Uses simulated time (no wall-clock delays)
- Time advances only when events occur
- Events are processed instantaneously
- Simulation runs as fast as possible
- No time.sleep() calls anywhere
"""

import sys
import os
import uuid
import signal
import random
import heapq
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod

# Import visualization if available
try:
    from visualize_results import JobSchedulingVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  Visualization not available. Install matplotlib, seaborn, pandas, numpy")

class Priority(Enum):
    """Job priority levels"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class EventType(Enum):
    """Types of simulation events"""
    JOB_ARRIVAL = "job_arrival"
    JOB_START = "job_start"
    JOB_COMPLETE = "job_complete"
    JOB_FAIL = "job_fail"
    AGENT_AVAILABLE = "agent_available"
    SYSTEM_SHUTDOWN = "system_shutdown"

@dataclass
class SimulationEvent:
    """A discrete event in the simulation"""
    time: float  # Simulation time (not wall clock)
    event_type: EventType
    data: Dict[str, Any]
    priority: int = 0  # For event ordering (lower = higher priority)
    
    def __lt__(self, other):
        """For heap ordering - events with same time ordered by priority"""
        if self.time == other.time:
            return self.priority < other.priority
        return self.time < other.time

class SimulationClock:
    """Manages simulated time - advances only when events occur"""
    
    def __init__(self):
        self.current_time = 0.0
        self.event_count = 0
    
    def advance_to(self, time: float):
        """Advance simulation time to specified time"""
        if time >= self.current_time:
            self.current_time = time
            self.event_count += 1
        else:
            raise ValueError(f"Cannot go backwards in time: {time} < {self.current_time}")
    
    def now(self) -> float:
        """Get current simulation time"""
        return self.current_time
    
    def reset(self):
        """Reset simulation time"""
        self.current_time = 0.0
        self.event_count = 0

@dataclass
class HPC_Resource:
    """HPC Resource definition"""
    name: str
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    cost_per_hour: float = 10.0
    
    def can_handle_job(self, requirements: Dict[str, Any]) -> tuple[bool, float]:
        """Check if resource can handle job requirements"""
        cpu_req = requirements.get('cpu', 0)
        memory_req = requirements.get('memory', 0)
        gpu_req = requirements.get('gpu', 0)
        
        if (cpu_req <= self.cpu_cores and 
            memory_req <= self.memory_gb and 
            gpu_req <= self.gpu_count):
            
            # Calculate match score (0-1, higher is better)
            cpu_score = min(1.0, cpu_req / max(1, self.cpu_cores))
            memory_score = min(1.0, memory_req / max(1, self.memory_gb))
            gpu_score = min(1.0, gpu_req / max(1, self.gpu_count)) if self.gpu_count > 0 else (1.0 if gpu_req == 0 else 0.0)
            
            # Weighted average score
            score = (cpu_score * 0.4 + memory_score * 0.3 + gpu_score * 0.3)
            return True, score
        
        return False, 0.0

class DiscreteEventJob:
    """Job class for true discrete event simulation"""
    
    def __init__(self, job_id: str, name: str, priority: Priority, resource_requirements: Dict[str, Any], 
                 expected_duration: float = None, workload_complexity: float = 1.0):
        self.job_id = job_id
        self.id = job_id  # For compatibility
        self.name = name
        self.priority = priority
        self.resource_requirements = resource_requirements
        self.expected_duration = expected_duration  # Base duration in simulation time units
        self.workload_complexity = workload_complexity  # Complexity multiplier
        self.arrival_time = 0.0  # Simulation time
        self.status = "pending"
        self.assigned_agent = None
        self.start_time = None
        self.completion_time = None
        self.actual_duration = 0.0
        self.retry_count = 0
    
    def calculate_execution_time(self, agent_resource: HPC_Resource) -> float:
        """Calculate realistic execution time in simulation time units"""
        if self.expected_duration is not None:
            base_time = self.expected_duration
        else:
            # Calculate base time from resource requirements
            cpu_req = self.resource_requirements.get('cpu', 1)
            memory_req = self.resource_requirements.get('memory', 1)
            gpu_req = self.resource_requirements.get('gpu', 0)
            
            base_time = (cpu_req * 0.5) + (memory_req * 0.01) + (gpu_req * 2.0)
            base_time = max(1.0, base_time)
        
        # Apply workload complexity
        adjusted_time = base_time * self.workload_complexity
        
        # Resource efficiency factor
        cpu_efficiency = min(2.0, agent_resource.cpu_cores / max(1, self.resource_requirements.get('cpu', 1)))
        memory_efficiency = min(2.0, agent_resource.memory_gb / max(1, self.resource_requirements.get('memory', 1)))
        
        if self.resource_requirements.get('gpu', 0) > 0 and agent_resource.gpu_count > 0:
            gpu_efficiency = min(3.0, agent_resource.gpu_count / self.resource_requirements.get('gpu', 1))
        else:
            gpu_efficiency = 1.0
        
        overall_efficiency = (cpu_efficiency * 0.4 + memory_efficiency * 0.3 + gpu_efficiency * 0.3)
        final_time = adjusted_time / max(0.5, overall_efficiency)
        
        # Add randomness (±20%)
        variance = random.uniform(0.8, 1.2)
        final_time *= variance
        
        return max(0.1, final_time)  # Minimum 0.1 simulation time units

class DiscreteEventScheduler:
    """True discrete event scheduler"""
    
    def __init__(self, simulation):
        self.simulation = simulation
        self.pending_jobs: List[DiscreteEventJob] = []
        self.running_jobs: Dict[str, DiscreteEventJob] = {}
        self.completed_jobs: List[DiscreteEventJob] = []
        self.failed_jobs: List[DiscreteEventJob] = []
        
    def schedule_job(self, job: DiscreteEventJob):
        """Schedule a job for execution"""
        self.pending_jobs.append(job)
        print(f"📋 Scheduler received job {job.id} at t={self.simulation.clock.now():.2f}")
        
        # Try to assign immediately
        self._try_assign_jobs()
    
    def _try_assign_jobs(self):
        """Try to assign pending jobs to available agents"""
        if not self.pending_jobs:
            return
        
        # Sort jobs by priority
        self.pending_jobs.sort(key=lambda j: (j.priority.value, j.arrival_time))
        
        assigned_jobs = []
        unassigned_jobs = []
        
        for job in self.pending_jobs:
            best_agent = self._find_best_agent(job)
            if best_agent:
                self._assign_job_to_agent(job, best_agent)
                assigned_jobs.append(job)
            else:
                unassigned_jobs.append(job)
                # Job remains in pending queue for future assignment
        
        # Update pending jobs list to only contain unassigned jobs
        self.pending_jobs = unassigned_jobs
        
        if assigned_jobs:
            print(f"🎯 Assigned {len(assigned_jobs)} jobs at t={self.simulation.clock.now():.2f}")
        
        if unassigned_jobs:
            print(f"⏳ {len(unassigned_jobs)} jobs remain queued (waiting for resources)")
    
    def _find_best_agent(self, job: DiscreteEventJob) -> Optional['DiscreteEventAgent']:
        """Find the best available agent for the job"""
        best_agent = None
        best_score = -1
        
        for agent in self.simulation.agents.values():
            if agent.is_available():
                can_handle, score = agent.resource.can_handle_job(job.resource_requirements)
                if can_handle and score > best_score:
                    best_score = score
                    best_agent = agent
        
        return best_agent
    
    def _assign_job_to_agent(self, job: DiscreteEventJob, agent: 'DiscreteEventAgent'):
        """Assign a job to an agent"""
        job.assigned_agent = agent.agent_id
        job.status = "assigned"
        
        # Schedule job start event
        start_event = SimulationEvent(
            time=self.simulation.clock.now(),
            event_type=EventType.JOB_START,
            data={'job_id': job.id, 'agent_id': agent.agent_id},
            priority=1
        )
        self.simulation.schedule_event(start_event)
    
    def handle_job_completion(self, job: DiscreteEventJob):
        """Handle job completion"""
        if job.id in self.running_jobs:
            del self.running_jobs[job.id]
        
        if job.status == "completed":
            self.completed_jobs.append(job)
            print(f"✅ Job {job.id} completed at t={self.simulation.clock.now():.2f} (duration: {job.actual_duration:.2f})")
        else:
            print(f"💥 Job {job.id} failed at t={self.simulation.clock.now():.2f}")
            
            # Always retry failed jobs by putting them back in the pending queue
            # This simulates a real HPC system where jobs are requeued when resources become available
            job.retry_count += 1
            job.status = "pending"
            job.assigned_agent = None
            self.pending_jobs.append(job)
            print(f"🔄 Requeuing job {job.id} for retry (attempt {job.retry_count + 1})")
        
        # Try to assign more jobs when resources become available
        self._try_assign_jobs()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            'pending_jobs': len(self.pending_jobs),
            'running_jobs': len(self.running_jobs),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'total_jobs': len(self.pending_jobs) + len(self.running_jobs) + len(self.completed_jobs) + len(self.failed_jobs)
        }

class DiscreteEventAgent:
    """Resource agent for true discrete event simulation"""
    
    def __init__(self, agent_id: str, resource: HPC_Resource, simulation, failure_rate: float = 0.1):
        self.agent_id = agent_id
        self.resource = resource
        self.simulation = simulation
        self.failure_rate = failure_rate
        
        # Agent state
        self.running_jobs: Dict[str, DiscreteEventJob] = {}
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.available_cpu = resource.cpu_cores
        self.available_memory = resource.memory_gb
        self.available_gpu = resource.gpu_count
        self.total_busy_time = 0.0
        
    def is_available(self) -> bool:
        """Check if agent has available resources"""
        return self.available_cpu > 0 and self.available_memory > 0 and (
            self.resource.gpu_count == 0 or self.available_gpu > 0
        )
    
    def can_handle_job(self, job: DiscreteEventJob) -> bool:
        """Check if agent can handle the job with current available resources"""
        cpu_req = job.resource_requirements.get('cpu', 0)
        memory_req = job.resource_requirements.get('memory', 0)
        gpu_req = job.resource_requirements.get('gpu', 0)
        
        return (cpu_req <= self.available_cpu and 
                memory_req <= self.available_memory and 
                gpu_req <= self.available_gpu)
    
    def start_job(self, job: DiscreteEventJob):
        """Start executing a job"""
        if not self.can_handle_job(job):
            print(f"❌ Agent {self.agent_id} cannot handle job {job.id} - insufficient resources")
            return False
        
        # Reserve resources
        cpu_req = job.resource_requirements.get('cpu', 0)
        memory_req = job.resource_requirements.get('memory', 0)
        gpu_req = job.resource_requirements.get('gpu', 0)
        
        self.available_cpu -= cpu_req
        self.available_memory -= memory_req
        self.available_gpu -= gpu_req
        
        # Start job
        job.start_time = self.simulation.clock.now()
        job.status = "running"
        self.running_jobs[job.id] = job
        self.simulation.scheduler.running_jobs[job.id] = job
        
        # Calculate execution time
        execution_time = job.calculate_execution_time(self.resource)
        job.actual_duration = execution_time
        
        print(f"▶️  Agent {self.agent_id} started job {job.id} at t={job.start_time:.2f} (duration: {execution_time:.2f})")
        
        # Schedule completion/failure event
        completion_time = job.start_time + execution_time
        
        # Determine if job will fail
        will_fail = random.random() < self.failure_rate
        event_type = EventType.JOB_FAIL if will_fail else EventType.JOB_COMPLETE
        
        completion_event = SimulationEvent(
            time=completion_time,
            event_type=event_type,
            data={'job_id': job.id, 'agent_id': self.agent_id},
            priority=2
        )
        self.simulation.schedule_event(completion_event)
        
        return True
    
    def complete_job(self, job: DiscreteEventJob, success: bool = True):
        """Complete a job (success or failure)"""
        if job.id not in self.running_jobs:
            return
        
        # Release resources
        cpu_req = job.resource_requirements.get('cpu', 0)
        memory_req = job.resource_requirements.get('memory', 0)
        gpu_req = job.resource_requirements.get('gpu', 0)
        
        self.available_cpu += cpu_req
        self.available_memory += memory_req
        self.available_gpu += gpu_req
        
        # Update job status
        job.completion_time = self.simulation.clock.now()
        job.status = "completed" if success else "failed"
        
        if success:
            self.completed_jobs += 1
        else:
            self.failed_jobs += 1
        
        # Update busy time
        if job.start_time is not None:
            self.total_busy_time += (job.completion_time - job.start_time)
        
        # Remove from running jobs
        del self.running_jobs[job.id]
        
        # Notify scheduler
        self.simulation.scheduler.handle_job_completion(job)
        
        return job
    
    def get_utilization(self) -> float:
        """Get current utilization percentage"""
        if self.simulation.clock.now() == 0:
            return 0.0
        return (self.total_busy_time / self.simulation.clock.now()) * 100

class TrueDiscreteEventSimulation:
    """Main simulation engine for true discrete event scheduling"""
    
    def __init__(self, visualizer=None):
        self.clock = SimulationClock()
        self.event_queue: List[SimulationEvent] = []
        self.agents: Dict[str, DiscreteEventAgent] = {}
        self.scheduler = DiscreteEventScheduler(self)
        self.visualizer = visualizer
        self.running = False
        self.job_registry: Dict[str, DiscreteEventJob] = {}  # Store all jobs by ID
        self.stats = {
            'events_processed': 0,
            'jobs_submitted': 0,
            'simulation_start_time': None,
            'simulation_end_time': None
        }
    
    def add_agent(self, agent: DiscreteEventAgent):
        """Add an agent to the simulation"""
        self.agents[agent.agent_id] = agent
    
    def schedule_event(self, event: SimulationEvent):
        """Schedule an event for future processing"""
        heapq.heappush(self.event_queue, event)
    
    def submit_job(self, job: DiscreteEventJob, arrival_time: float = None):
        """Submit a job to the simulation"""
        if arrival_time is None:
            arrival_time = self.clock.now()
        
        job.arrival_time = arrival_time
        self.stats['jobs_submitted'] += 1
        
        # Store job in registry for later retrieval
        self.job_registry[job.id] = job
        
        # Schedule job arrival event
        arrival_event = SimulationEvent(
            time=arrival_time,
            event_type=EventType.JOB_ARRIVAL,
            data={'job_id': job.id},
            priority=0
        )
        self.schedule_event(arrival_event)
    
    def run(self, max_simulation_time: float = 1000.0):
        """Run the simulation until completion or timeout"""
        self.running = True
        self.stats['simulation_start_time'] = datetime.now()
        
        print(f"\n🚀 Starting TRUE discrete event simulation...")
        print(f"📊 {len(self.agents)} agents, {len(self.event_queue)} initial events")
        print(f"⚡ Time advances only when events occur - no waiting!")
        
        while self.running and self.event_queue and self.clock.now() < max_simulation_time:
            # Get next event
            event = heapq.heappop(self.event_queue)
            
            # Advance simulation time to event time
            self.clock.advance_to(event.time)
            
            # Process event
            self._process_event(event)
            self.stats['events_processed'] += 1
            
            # Check termination conditions
            if self._should_terminate():
                break
        
        self.stats['simulation_end_time'] = datetime.now()
        self.running = False
        
        wall_time = (self.stats['simulation_end_time'] - self.stats['simulation_start_time']).total_seconds()
        print(f"\n🎉 Simulation completed!")
        print(f"⏱️  Simulation time: {self.clock.now():.2f} time units")
        print(f"⚡ Wall clock time: {wall_time:.3f} seconds")
        print(f"🔥 Speed ratio: {self.clock.now()/wall_time:.1f}x faster than real-time")
        print(f"📈 Events processed: {self.stats['events_processed']}")
    
    def _process_event(self, event: SimulationEvent):
        """Process a single simulation event"""
        if event.event_type == EventType.JOB_ARRIVAL:
            self._handle_job_arrival(event)
        elif event.event_type == EventType.JOB_START:
            self._handle_job_start(event)
        elif event.event_type == EventType.JOB_COMPLETE:
            self._handle_job_completion(event, success=True)
        elif event.event_type == EventType.JOB_FAIL:
            self._handle_job_completion(event, success=False)
        elif event.event_type == EventType.SYSTEM_SHUTDOWN:
            self.running = False
    
    def _handle_job_arrival(self, event: SimulationEvent):
        """Handle job arrival event"""
        job_id = event.data['job_id']
        # Find the job from the registry
        job = self.job_registry.get(job_id)
        
        if job:
            self.scheduler.schedule_job(job)
        else:
            print(f"⚠️  Could not find job {job_id} for arrival event")
    
    def _handle_job_start(self, event: SimulationEvent):
        """Handle job start event"""
        job_id = event.data['job_id']
        agent_id = event.data['agent_id']
        
        agent = self.agents[agent_id]
        job = self._find_job_by_id(job_id)
        
        if job and agent:
            agent.start_job(job)
    
    def _handle_job_completion(self, event: SimulationEvent, success: bool):
        """Handle job completion event"""
        job_id = event.data['job_id']
        agent_id = event.data['agent_id']
        
        agent = self.agents[agent_id]
        job = agent.running_jobs.get(job_id)
        
        if job:
            agent.complete_job(job, success)
    
    def _find_job_by_id(self, job_id: str) -> Optional[DiscreteEventJob]:
        """Find a job by ID across all collections"""
        # First check the job registry (most reliable source)
        job = self.job_registry.get(job_id)
        if job:
            return job
        
        # Fallback: Check scheduler collections
        for job in self.scheduler.pending_jobs:
            if job.id == job_id:
                return job
        
        for job in self.scheduler.running_jobs.values():
            if job.id == job_id:
                return job
        
        for job in self.scheduler.completed_jobs:
            if job.id == job_id:
                return job
        
        for job in self.scheduler.failed_jobs:
            if job.id == job_id:
                return job
        
        return None
    
    def _get_all_jobs(self) -> List[DiscreteEventJob]:
        """Get all jobs in the simulation"""
        all_jobs = []
        all_jobs.extend(self.scheduler.pending_jobs)
        all_jobs.extend(self.scheduler.running_jobs.values())
        all_jobs.extend(self.scheduler.completed_jobs)
        all_jobs.extend(self.scheduler.failed_jobs)
        return all_jobs
    
    def _should_terminate(self) -> bool:
        """Check if simulation should terminate"""
        # Get current scheduler statistics
        scheduler_stats = self.scheduler.get_stats()
        
        # Don't terminate if there are still running jobs, even if event queue is empty
        # This ensures all running jobs complete before termination
        if scheduler_stats['running_jobs'] > 0:
            return False
        
        # Don't terminate if there are pending jobs waiting to be scheduled
        if scheduler_stats['pending_jobs'] > 0:
            return False
            
        # Check if there are events scheduled for the current time that haven't been processed yet
        current_time = self.clock.now()
        events_at_current_time = [event for event in self.event_queue if event.time == current_time]
        if events_at_current_time:
            return False
        
        # Only terminate when:
        # 1. Event queue is empty OR all remaining events are in the future
        # 2. No pending jobs
        # 3. No running jobs  
        # 4. No events scheduled at current time
        return len(self.event_queue) == 0
    
    def get_final_stats(self) -> Dict[str, Any]:
        """Get final simulation statistics"""
        scheduler_stats = self.scheduler.get_stats()
        
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            agent_stats[agent_id] = {
                'completed_jobs': agent.completed_jobs,
                'failed_jobs': agent.failed_jobs,
                'utilization': agent.get_utilization(),
                'total_busy_time': agent.total_busy_time
            }
        
        wall_time = 0.0
        if self.stats['simulation_start_time'] and self.stats['simulation_end_time']:
            wall_time = (self.stats['simulation_end_time'] - self.stats['simulation_start_time']).total_seconds()
        
        return {
            'simulation_time': self.clock.now(),
            'wall_clock_time': wall_time,
            'speed_ratio': self.clock.now() / wall_time if wall_time > 0 else float('inf'),
            'events_processed': self.stats['events_processed'],
            'jobs_submitted': self.stats['jobs_submitted'],
            'scheduler_stats': scheduler_stats,
            'agent_stats': agent_stats
        }

def create_realistic_hpc_workload() -> List[DiscreteEventJob]:
    """Create a realistic HPC workload with staggered arrivals"""
    jobs = []
    
    # Job categories with realistic timing and arrival patterns
    job_categories = [
        # (name, priority, requirements, base_duration, complexity, arrival_time_range)
        
        # Critical system jobs - arrive immediately
        ("System Health Check", Priority.CRITICAL, {"cpu": 1, "memory": 1, "gpu": 0}, 2.0, 0.5, (0.0, 0.1)),
        
        # High priority development jobs - arrive early
        ("Unit Test Suite", Priority.HIGH, {"cpu": 1, "memory": 2, "gpu": 0}, 3.0, 0.8, (0.2, 1.0)),
        ("Debug Session", Priority.HIGH, {"cpu": 1, "memory": 2, "gpu": 0}, 12.0, 1.5, (0.5, 1.5)),
        ("Quantum Chemistry", Priority.HIGH, {"cpu": 8, "memory": 32, "gpu": 0}, 25.0, 1.4, (1.0, 2.0)),
        ("Deep Learning Training", Priority.HIGH, {"cpu": 8, "memory": 128, "gpu": 4}, 120.0, 2.5, (1.5, 3.0)),
        ("Computer Vision Model", Priority.HIGH, {"cpu": 6, "memory": 64, "gpu": 2}, 80.0, 2.0, (2.0, 3.5)),
        ("Financial Risk Modeling", Priority.HIGH, {"cpu": 16, "memory": 256, "gpu": 0}, 65.0, 1.9, (2.5, 4.0)),
        
        # Medium priority jobs - regular workload
        ("Code Compilation", Priority.MEDIUM, {"cpu": 2, "memory": 4, "gpu": 0}, 8.0, 1.2, (3.0, 8.0)),
        ("Integration Test", Priority.MEDIUM, {"cpu": 1, "memory": 1, "gpu": 0}, 5.0, 0.9, (4.0, 9.0)),
        ("Performance Profiling", Priority.MEDIUM, {"cpu": 2, "memory": 4, "gpu": 0}, 15.0, 1.3, (5.0, 10.0)),
        ("Climate Simulation", Priority.MEDIUM, {"cpu": 16, "memory": 64, "gpu": 0}, 45.0, 1.8, (6.0, 12.0)),
        ("Molecular Dynamics", Priority.MEDIUM, {"cpu": 12, "memory": 48, "gpu": 0}, 35.0, 1.6, (7.0, 13.0)),
        ("CFD Analysis", Priority.MEDIUM, {"cpu": 16, "memory": 80, "gpu": 0}, 40.0, 1.7, (8.0, 14.0)),
        ("Protein Folding", Priority.MEDIUM, {"cpu": 24, "memory": 96, "gpu": 0}, 50.0, 2.0, (9.0, 15.0)),
        ("Weather Modeling", Priority.MEDIUM, {"cpu": 28, "memory": 100, "gpu": 0}, 60.0, 1.9, (10.0, 16.0)),
        ("Seismic Processing", Priority.MEDIUM, {"cpu": 16, "memory": 64, "gpu": 0}, 30.0, 1.5, (11.0, 17.0)),
        ("Neural Network Inference", Priority.MEDIUM, {"cpu": 4, "memory": 32, "gpu": 1}, 8.0, 0.7, (12.0, 18.0)),
        ("NLP Transformer Training", Priority.MEDIUM, {"cpu": 16, "memory": 256, "gpu": 4}, 150.0, 2.8, (13.0, 19.0)),
        ("Reinforcement Learning", Priority.MEDIUM, {"cpu": 4, "memory": 16, "gpu": 1}, 90.0, 2.2, (14.0, 20.0)),
        ("Genomics Analysis", Priority.MEDIUM, {"cpu": 8, "memory": 512, "gpu": 0}, 75.0, 2.1, (15.0, 21.0)),
        ("Large Dataset Processing", Priority.MEDIUM, {"cpu": 4, "memory": 128, "gpu": 0}, 55.0, 1.7, (16.0, 22.0)),
        ("Statistical Analysis", Priority.MEDIUM, {"cpu": 6, "memory": 64, "gpu": 0}, 35.0, 1.4, (17.0, 23.0)),
        
        # Low priority batch jobs - arrive later
        ("Model Hyperparameter Tuning", Priority.LOW, {"cpu": 2, "memory": 8, "gpu": 1}, 45.0, 1.3, (20.0, 30.0)),
        ("Nightly Data Backup", Priority.LOW, {"cpu": 1, "memory": 4, "gpu": 0}, 20.0, 0.9, (25.0, 35.0)),
        ("Log File Processing", Priority.LOW, {"cpu": 2, "memory": 8, "gpu": 0}, 25.0, 1.1, (30.0, 40.0)),
        ("Report Generation", Priority.LOW, {"cpu": 1, "memory": 2, "gpu": 0}, 15.0, 0.8, (35.0, 45.0)),
    ]
    
    # Create jobs with random arrival times within ranges
    for name, priority, req, duration, complexity, (min_arrival, max_arrival) in job_categories:
        arrival_time = random.uniform(min_arrival, max_arrival)
        job = DiscreteEventJob(
            job_id=str(uuid.uuid4()),
            name=name,
            priority=priority,
            resource_requirements=req,
            expected_duration=duration,
            workload_complexity=complexity
        )
        jobs.append((job, arrival_time))
    
    return jobs

def run_true_discrete_event_demo():
    """Run the true discrete event scheduling demo"""
    
    print("\n" + "="*90)
    print(" TRUE DISCRETE EVENT SCHEDULING DEMO - Zero Time Delays")
    print("="*90)
    
    # Initialize visualizer
    visualizer = None
    if VISUALIZATION_AVAILABLE:
        print("\n🎨 Initializing visualization...")
        visualizer = JobSchedulingVisualizer(save_plots=True, output_dir="true_discrete_event_plots")
        print("✅ Visualization system ready!")
    else:
        print("\n⚠️  Running without visualization")
    
    print("\n🎯 This TRUE discrete event demo features:")
    print("  • Simulated time (no wall-clock delays)")
    print("  • Time advances only when events occur")
    print("  • Events processed instantaneously")
    print("  • No time.sleep() calls anywhere")
    print("  • Simulation runs as fast as possible")
    
    # Create simulation
    simulation = TrueDiscreteEventSimulation(visualizer)
    
    # Create realistic HPC cluster
    resources = [
        ("small-cpu-1", HPC_Resource("Small CPU Alpha", 8, 32, 0, 3.5), 0.05),
        ("small-cpu-2", HPC_Resource("Small CPU Beta", 12, 48, 0, 4.2), 0.08),
        ("small-gpu-1", HPC_Resource("Small GPU", 4, 64, 2, 12.0), 0.12),
        ("medium-cpu-1", HPC_Resource("Medium CPU Gamma", 32, 128, 0, 8.5), 0.10),
        ("medium-cpu-2", HPC_Resource("Medium CPU Delta", 28, 112, 0, 7.8), 0.12),
        ("medium-gpu-1", HPC_Resource("Medium GPU", 16, 256, 4, 25.0), 0.18),
        ("medium-hybrid-1", HPC_Resource("Medium Hybrid", 24, 192, 2, 18.5), 0.15),
        ("large-cpu-1", HPC_Resource("Large CPU Epsilon", 64, 512, 0, 15.0), 0.08),
        ("large-cpu-2", HPC_Resource("Large CPU Zeta", 72, 576, 0, 16.5), 0.10),
        ("large-gpu-1", HPC_Resource("Large GPU", 32, 1024, 8, 45.0), 0.20),
        ("large-hybrid-1", HPC_Resource("Large Hybrid", 48, 768, 4, 32.0), 0.15),
        ("large-memory-1", HPC_Resource("Large Memory", 16, 2048, 0, 35.0), 0.06)
    ]
    
    # Add agents to simulation
    for agent_id, resource, failure_rate in resources:
        agent = DiscreteEventAgent(agent_id, resource, simulation, failure_rate)
        simulation.add_agent(agent)
    
    print(f"\n📋 Simulation Configuration ({len(simulation.agents)} agents):")
    for agent_id, agent in simulation.agents.items():
        resource = agent.resource
        print(f"  • {agent_id}: {resource.cpu_cores}CPU/{resource.memory_gb}GB/{resource.gpu_count}GPU "
              f"(failure: {agent.failure_rate*100:.0f}%)")
    
    # Create realistic workload
    job_arrival_pairs = create_realistic_hpc_workload()
    
    print(f"\n📤 Scheduling {len(job_arrival_pairs)} jobs with realistic arrival times...")
    
    # Submit all jobs to simulation
    for job, arrival_time in job_arrival_pairs:
        simulation.submit_job(job, arrival_time)
        print(f"  📋 {job.name} ({job.priority.name}) -> arrives at t={arrival_time:.2f}")
    
    # Run the simulation
    print(f"\n⚡ Starting simulation with {len(simulation.event_queue)} events...")
    simulation.run(max_simulation_time=500.0)
    
    # Get final statistics
    final_stats = simulation.get_final_stats()
    
    print(f"\n📊 Final TRUE Discrete Event Results:")
    print(f"  • Simulation time: {final_stats['simulation_time']:.2f} time units")
    print(f"  • Wall clock time: {final_stats['wall_clock_time']:.3f} seconds")
    print(f"  • Speed ratio: {final_stats['speed_ratio']:.1f}x faster than real-time")
    print(f"  • Events processed: {final_stats['events_processed']}")
    print(f"  • Jobs submitted: {final_stats['jobs_submitted']}")
    
    scheduler_stats = final_stats['scheduler_stats']
    print(f"  • Jobs completed: {scheduler_stats['completed_jobs']}")
    print(f"  • Jobs failed: {scheduler_stats['failed_jobs']}")
    print(f"  • Jobs pending: {scheduler_stats['pending_jobs']}")
    total_processed = scheduler_stats['completed_jobs'] + scheduler_stats['failed_jobs']
    if total_processed > 0:
        success_rate = (scheduler_stats['completed_jobs'] / total_processed) * 100
        print(f"  • Success rate: {success_rate:.1f}%")
    else:
        print(f"  • Success rate: N/A (no jobs completed or failed yet)")
    
    print(f"\n🤖 Agent Performance:")
    for agent_id, stats in final_stats['agent_stats'].items():
        if stats['completed_jobs'] > 0 or stats['failed_jobs'] > 0:
            efficiency = stats['completed_jobs'] / (stats['completed_jobs'] + stats['failed_jobs']) * 100
            print(f"  • {agent_id}: {stats['completed_jobs']}C/{stats['failed_jobs']}F "
                  f"(efficiency: {efficiency:.0f}%, utilization: {stats['utilization']:.1f}%)")
    
    print("\n" + "="*90)
    print(" Demonstrated: TRUE discrete event simulation with zero time delays")
    print("="*90)

def main():
    """Main entry point"""
    def signal_handler(signum, frame):
        print("\n\n🛑 Demo interrupted by user")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        run_true_discrete_event_demo()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n✅ Demo cleanup completed")

if __name__ == "__main__":
    main()

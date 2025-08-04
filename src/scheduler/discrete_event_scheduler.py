import threading
import heapq
from datetime import datetime, timedelta
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum

from ..communication.protocol import MessageBus, MessageType, MessagePriority, Message
from ..jobs.job import Job, JobStatus
from .job_pool import JobPool

class SchedulingEventType(Enum):
    """Types of scheduling events"""
    JOB_ARRIVAL = "job_arrival"
    JOB_COMPLETE = "job_complete"
    JOB_FAILED = "job_failed"
    RESOURCE_AVAILABLE = "resource_available"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    RESOURCE_FAILURE = "resource_failure"
    AGENT_HEARTBEAT = "agent_heartbeat"
    SCHEDULING_DECISION = "scheduling_decision"
    TIMEOUT_CHECK = "timeout_check"
    SYSTEM_SHUTDOWN = "system_shutdown"

@dataclass
class SchedulingEvent:
    """A discrete event in the scheduling system"""
    timestamp: datetime
    event_type: SchedulingEventType
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Lower number = higher priority
    
    def __lt__(self, other):
        # First compare by timestamp, then by priority
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.priority < other.priority

class DiscreteEventScheduler:
    """Pure discrete event scheduler - no time-based polling"""
    
    def __init__(self, message_bus: MessageBus, job_pool: JobPool):
        self.message_bus = message_bus
        self.job_pool = job_pool
        
        # Event queue (priority queue)
        self.event_queue: List[SchedulingEvent] = []
        self.event_lock = threading.Lock()
        
        # System state
        self.current_time = datetime.now()
        self.running = False
        self.available_agents: Dict[str, Dict] = {}
        self.busy_agents: Dict[str, Dict] = {}
        self.pending_offers: Dict[str, List[Dict]] = {}
        self.job_assignments: Dict[str, str] = {}  # job_id -> agent_id
        
        # Event handlers
        self.event_handlers: Dict[SchedulingEventType, Callable] = {
            SchedulingEventType.JOB_ARRIVAL: self._handle_job_arrival,
            SchedulingEventType.JOB_COMPLETE: self._handle_job_complete,
            SchedulingEventType.JOB_FAILED: self._handle_job_failed,
            SchedulingEventType.RESOURCE_AVAILABLE: self._handle_resource_available,
            SchedulingEventType.RESOURCE_UNAVAILABLE: self._handle_resource_unavailable,
            SchedulingEventType.RESOURCE_FAILURE: self._handle_resource_failure,
            SchedulingEventType.AGENT_HEARTBEAT: self._handle_agent_heartbeat,
            SchedulingEventType.SCHEDULING_DECISION: self._handle_scheduling_decision,
            SchedulingEventType.TIMEOUT_CHECK: self._handle_timeout_check,
            SchedulingEventType.SYSTEM_SHUTDOWN: self._handle_system_shutdown,
        }
        
        # Thread management
        self._worker_thread = None
        self._stop_event = threading.Event()
        
        # Subscribe to message bus
        self.message_bus.subscribe('scheduler', self._on_message)
        print("🎯 Discrete Event Scheduler initialized")
    
    def schedule_event(self, event: SchedulingEvent):
        """Schedule a new event"""
        with self.event_lock:
            heapq.heappush(self.event_queue, event)
            # Wake up the worker thread if it's waiting
            self._stop_event.set()
            self._stop_event.clear()
    
    def start(self):
        """Start the discrete event scheduler"""
        if self.running:
            return
            
        self.running = True
        self._worker_thread = threading.Thread(target=self._run, daemon=True)
        self._worker_thread.start()
        
        # Schedule initial timeout check
        self._schedule_timeout_check()
        print("🚀 Discrete Event Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        if not self.running:
            return
            
        # Schedule shutdown event
        shutdown_event = SchedulingEvent(
            timestamp=datetime.now(),
            event_type=SchedulingEventType.SYSTEM_SHUTDOWN,
            priority=0  # High priority shutdown
        )
        self.schedule_event(shutdown_event)
        
        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        
        self.running = False
        print("🛑 Discrete Event Scheduler stopped")
    
    def _run(self):
        """Main event processing loop - purely event-driven"""
        while self.running:
            event = None
            
            with self.event_lock:
                if self.event_queue:
                    event = heapq.heappop(self.event_queue)
                    self.current_time = max(self.current_time, event.timestamp)
            
            if event:
                self._process_event(event)
            else:
                # No events - wait briefly to prevent busy waiting
                self._stop_event.wait(0.001)  # 1ms wait
    
    def _process_event(self, event: SchedulingEvent):
        """Process a single event"""
        try:
            handler = self.event_handlers.get(event.event_type)
            if handler:
                handler(event)
            else:
                print(f"❌ No handler for event type: {event.event_type}")
        except Exception as e:
            print(f"💥 Error processing event {event.event_type}: {e}")
    
    def _on_message(self, message: Message):
        """Convert messages to scheduling events"""
        event_data = {
            'message': message,
            'sender_id': message.sender_id,
            'payload': message.payload
        }
        
        if message.message_type == MessageType.JOB_COMPLETE:
            event = SchedulingEvent(
                timestamp=datetime.now(),
                event_type=SchedulingEventType.JOB_COMPLETE,
                data=event_data,
                priority=1
            )
            self.schedule_event(event)
            
        elif message.message_type == MessageType.RESOURCE_OFFER:
            # Job arrival triggers immediate scheduling decision
            job_id = message.payload.get('job_id')
            if job_id:
                arrival_event = SchedulingEvent(
                    timestamp=datetime.now(),
                    event_type=SchedulingEventType.JOB_ARRIVAL,
                    data={**event_data, 'job_id': job_id},
                    priority=2
                )
                self.schedule_event(arrival_event)
        
        elif message.message_type == MessageType.NEGOTIATE:
            # Check if this is a job rejection, failure, or resource availability
            payload = message.payload
            if payload.get('response') == 'rejected':
                # Handle job rejection
                event = SchedulingEvent(
                    timestamp=datetime.now(),
                    event_type=SchedulingEventType.JOB_FAILED,
                    data={**event_data, 'job_id': payload.get('job_id'), 'reason': 'rejected'},
                    priority=2
                )
                self.schedule_event(event)
            elif payload.get('response') == 'failed':
                # Handle job execution failure
                event = SchedulingEvent(
                    timestamp=datetime.now(),
                    event_type=SchedulingEventType.JOB_FAILED,
                    data={**event_data, 'job_id': payload.get('job_id'), 'reason': 'execution_failure', 'agent_id': message.sender_id},
                    priority=1  # Higher priority for actual failures
                )
                self.schedule_event(event)
            else:
                # Handle as resource availability change
                event = SchedulingEvent(
                    timestamp=datetime.now(),
                    event_type=SchedulingEventType.RESOURCE_AVAILABLE,
                    data=event_data,
                    priority=3
                )
                self.schedule_event(event)
    
    def _handle_job_arrival(self, event: SchedulingEvent):
        """Handle new job arrival or resource offer"""
        message = event.data.get('message')
        if not message:
            return
        
        job_id = message.payload.get('job_id')
        sender_id = message.sender_id
        offer = message.payload.get('resource', {})
        
        if job_id:
            print(f"📨 Job offer for {job_id} from {sender_id}")
            
            # Store the offer
            if job_id not in self.pending_offers:
                self.pending_offers[job_id] = []
            
            offer_data = {
                'agent_id': sender_id,
                'score': offer.get('score', 0),
                'offer': offer,
                'timestamp': event.timestamp
            }
            self.pending_offers[job_id].append(offer_data)
            
            # Trigger immediate scheduling decision
            decision_event = SchedulingEvent(
                timestamp=event.timestamp,
                event_type=SchedulingEventType.SCHEDULING_DECISION,
                data={'job_id': job_id, 'trigger': 'job_arrival'},
                priority=1
            )
            self.schedule_event(decision_event)
    
    def _handle_job_complete(self, event: SchedulingEvent):
        """Handle job completion"""
        message = event.data.get('message')
        if not message:
            return
        
        job_id = message.payload.get('job_id')
        status = message.payload.get('status')
        agent_id = message.sender_id
        
        if job_id and status:
            print(f"✅ Job {job_id} completed with status: {status}")
            
            # Update job pool
            try:
                job_status = JobStatus(status)
                self.job_pool.update_job_status(job_id, job_status)
            except ValueError:
                print(f"❌ Invalid job status: {status}")
            
            # Free up the agent
            if agent_id in self.busy_agents:
                agent_info = self.busy_agents.pop(agent_id)
                self.available_agents[agent_id] = agent_info
                
                # Trigger scheduling decision for pending jobs
                if self.job_pool.has_pending_jobs():
                    decision_event = SchedulingEvent(
                        timestamp=event.timestamp,
                        event_type=SchedulingEventType.SCHEDULING_DECISION,
                        data={'trigger': 'job_complete', 'freed_agent': agent_id},
                        priority=1
                    )
                    self.schedule_event(decision_event)
            
            # Clean up job assignment
            if job_id in self.job_assignments:
                del self.job_assignments[job_id]
    
    def _handle_job_failed(self, event: SchedulingEvent):
        """Handle job failure"""
        job_id = event.data.get('job_id')
        agent_id = event.data.get('agent_id')
        reason = event.data.get('reason', 'unknown')
        
        if job_id:
            if reason == 'rejected':
                print(f"⚠️ Job {job_id} rejected by agent, returning to pending")
                # Return job to pending status for rescheduling
                self.job_pool.update_job_status(job_id, JobStatus.PENDING)
                
                # Remove from job assignments if it was assigned
                if job_id in self.job_assignments:
                    del self.job_assignments[job_id]
                
                # Trigger immediate rescheduling decision
                decision_event = SchedulingEvent(
                    timestamp=event.timestamp,
                    event_type=SchedulingEventType.SCHEDULING_DECISION,
                    data={'trigger': 'job_rejected', 'job_id': job_id},
                    priority=1
                )
                self.schedule_event(decision_event)
            elif reason == 'execution_failure':
                print(f"💥 Job {job_id} failed during execution, returning to pending for retry")
                # Return job to pending status for retry
                self.job_pool.update_job_status(job_id, JobStatus.PENDING)
                
                # Remove from job assignments
                if job_id in self.job_assignments:
                    del self.job_assignments[job_id]
                
                # Trigger immediate rescheduling decision
                decision_event = SchedulingEvent(
                    timestamp=event.timestamp,
                    event_type=SchedulingEventType.SCHEDULING_DECISION,
                    data={'trigger': 'job_failed_retry', 'job_id': job_id},
                    priority=1
                )
                self.schedule_event(decision_event)
            else:
                print(f"💥 Job {job_id} failed permanently")
                
                # Update job status to failed for permanent failures
                self.job_pool.update_job_status(job_id, JobStatus.FAILED)
            
            # Free up the agent if it was busy
            if agent_id and agent_id in self.busy_agents:
                agent_info = self.busy_agents.pop(agent_id)
                self.available_agents[agent_id] = agent_info
                
                # Trigger rescheduling for remaining jobs
                if self.job_pool.has_pending_jobs():
                    decision_event = SchedulingEvent(
                        timestamp=event.timestamp,
                        event_type=SchedulingEventType.SCHEDULING_DECISION,
                        data={'trigger': 'job_failed', 'freed_agent': agent_id},
                        priority=1
                    )
                    self.schedule_event(decision_event)
    
    def _handle_resource_available(self, event: SchedulingEvent):
        """Handle resource becoming available"""
        agent_id = event.data.get('sender_id')
        if agent_id:
            print(f"💻 Agent {agent_id} is available")
            
            # Move agent to available list
            if agent_id in self.busy_agents:
                agent_info = self.busy_agents.pop(agent_id)
                # Update capabilities with current availability from the message
                agent_info.update({
                    'timestamp': event.timestamp,
                    'capabilities': event.data.get('payload', {}).get('capabilities', {})
                })
                self.available_agents[agent_id] = agent_info
            else:
                self.available_agents[agent_id] = {
                    'timestamp': event.timestamp,
                    'capabilities': event.data.get('payload', {}).get('capabilities', {})
                }
            
            # Trigger scheduling decision
            decision_event = SchedulingEvent(
                timestamp=event.timestamp,
                event_type=SchedulingEventType.SCHEDULING_DECISION,
                data={'trigger': 'resource_available', 'agent_id': agent_id},
                priority=2
            )
            self.schedule_event(decision_event)
    
    def _handle_resource_unavailable(self, event: SchedulingEvent):
        """Handle resource becoming unavailable"""
        agent_id = event.data.get('agent_id')
        if agent_id and agent_id in self.available_agents:
            print(f"❌ Agent {agent_id} is no longer available")
            del self.available_agents[agent_id]
    
    def _handle_resource_failure(self, event: SchedulingEvent):
        """Handle resource failure"""
        agent_id = event.data.get('agent_id')
        if agent_id:
            print(f"💥 Agent {agent_id} failed")
            
            # Remove from both available and busy lists
            self.available_agents.pop(agent_id, None)
            self.busy_agents.pop(agent_id, None)
            
            # Find and reschedule jobs assigned to this agent
            failed_jobs = [job_id for job_id, assigned_agent in self.job_assignments.items() 
                          if assigned_agent == agent_id]
            
            for job_id in failed_jobs:
                failure_event = SchedulingEvent(
                    timestamp=event.timestamp,
                    event_type=SchedulingEventType.JOB_FAILED,
                    data={'job_id': job_id, 'agent_id': agent_id},
                    priority=0  # High priority
                )
                self.schedule_event(failure_event)
    
    def _handle_agent_heartbeat(self, event: SchedulingEvent):
        """Handle agent heartbeat - update availability"""
        agent_id = event.data.get('agent_id')
        if agent_id:
            # Update agent's last seen time
            if agent_id in self.available_agents:
                self.available_agents[agent_id]['last_heartbeat'] = event.timestamp
            elif agent_id in self.busy_agents:
                self.busy_agents[agent_id]['last_heartbeat'] = event.timestamp
    
    def _handle_scheduling_decision(self, event: SchedulingEvent):
        """Make scheduling decisions - the core of the scheduler"""
        trigger = event.data.get('trigger', 'unknown')
        print(f"🧠 Making scheduling decision (triggered by: {trigger})")
        
        # Get pending jobs from job pool
        pending_jobs = self.job_pool.get_pending_jobs()
        
        if not pending_jobs or not self.available_agents:
            return
        
        # Sort jobs by priority and submission time
        sorted_jobs = sorted(pending_jobs, 
                           key=lambda j: (j.priority.value, j.submit_time))
        
        assignments_made = 0
        
        for job in sorted_jobs:
            if not self.available_agents:
                break  # No more available agents
            
            # Check if we have offers for this job
            job_offers = self.pending_offers.get(job.job_id, [])
            
            # If no offers exist, generate them from available agents
            if not job_offers:
                print(f"🔄 Generating new offers for job {job.job_id}")
                self._generate_offers_for_job(job, event.timestamp)
                job_offers = self.pending_offers.get(job.job_id, [])
            
            if not job_offers:
                print(f"❌ No capable agents found for job {job.job_id}")
                continue
            
            # Find best offer from available agents only
            available_offers = [offer for offer in job_offers 
                              if offer['agent_id'] in self.available_agents]
            
            if not available_offers:
                continue  # No available agents for this job
            
            best_offer = max(available_offers, key=lambda x: x.get('score', 0))
            best_agent_id = best_offer['agent_id']
            
            # Make the assignment
            self._assign_job_to_agent(job, best_agent_id, event.timestamp)
            assignments_made += 1
            
            # Clean up offers for this job
            del self.pending_offers[job.job_id]
        
        if assignments_made > 0:
            print(f"🎯 Made {assignments_made} job assignments")
    
    def _handle_timeout_check(self, event: SchedulingEvent):
        """Check for timeouts and stale states"""
        current_time = event.timestamp
        timeout_threshold = timedelta(minutes=5)
        
        # Check for stale offers
        stale_offers = []
        for job_id, offers in self.pending_offers.items():
            stale_offers.extend([
                (job_id, offer) for offer in offers 
                if current_time - offer['timestamp'] > timeout_threshold
            ])
        
        # Clean up stale offers
        for job_id, offer in stale_offers:
            self.pending_offers[job_id].remove(offer)
            if not self.pending_offers[job_id]:
                del self.pending_offers[job_id]
        
        if stale_offers:
            print(f"🧹 Cleaned up {len(stale_offers)} stale offers")
        
        # Schedule next timeout check
        self._schedule_timeout_check()
    
    def _handle_system_shutdown(self, event: SchedulingEvent):
        """Handle system shutdown"""
        print("🛑 Processing system shutdown event")
        self.running = False
    
    def _assign_job_to_agent(self, job: Job, agent_id: str, timestamp: datetime):
        """Assign a job to an agent"""
        print(f"🎯 Assigning job {job.job_id} to {agent_id}")
        
        # Move agent from available to busy
        if agent_id in self.available_agents:
            agent_info = self.available_agents.pop(agent_id)
            self.busy_agents[agent_id] = {
                **agent_info,
                'current_job': job.job_id,
                'job_start_time': timestamp
            }
        
        # Record assignment
        self.job_assignments[job.job_id] = agent_id
        
        # Send reservation message to agent
        reservation_message = Message(
            message_id="",
            message_type=MessageType.RESOURCE_RESERVATION,
            sender_id="scheduler",
            recipient_id=agent_id,
            timestamp=timestamp,
            priority=MessagePriority.HIGH,
            payload={"job": job.to_dict()}
        )
        self.message_bus.publish(reservation_message)
        
        # Update job status
        self.job_pool.update_job_status(job.job_id, JobStatus.RUNNING)
    
    def _generate_offers_for_job(self, job: Job, timestamp: datetime):
        """Generate offers for a job from all available agents"""
        if job.job_id not in self.pending_offers:
            self.pending_offers[job.job_id] = []
        
        for agent_id, agent_info in self.available_agents.items():
            capabilities = agent_info.get('capabilities', {})
            
            # Check if this agent can handle the job
            can_handle, score = self._can_agent_handle_job(capabilities, job.resource_requirements)
            
            if can_handle:
                offer_data = {
                    'agent_id': agent_id,
                    'score': score,
                    'offer': {
                        'score': score,
                        'capabilities': capabilities
                    },
                    'timestamp': timestamp
                }
                self.pending_offers[job.job_id].append(offer_data)
    
    def _can_agent_handle_job(self, capabilities: Dict[str, Any], requirements: Dict[str, Any]) -> tuple[bool, float]:
        """Check if an agent with given capabilities can handle job requirements"""
        cpu_req = requirements.get('cpu', 0)
        memory_req = requirements.get('memory', 0)
        gpu_req = requirements.get('gpu', 0)
        
        cpu_cores = capabilities.get('cpu_cores', 0)
        memory_gb = capabilities.get('memory_gb', 0)
        gpu_count = capabilities.get('gpu_count', 0)
        
        if (cpu_req <= cpu_cores and 
            memory_req <= memory_gb and 
            gpu_req <= gpu_count):
            
            # Calculate match score (0-1, higher is better)
            cpu_score = min(1.0, cpu_req / max(1, cpu_cores))
            memory_score = min(1.0, memory_req / max(1, memory_gb))
            gpu_score = min(1.0, gpu_req / max(1, gpu_count)) if gpu_count > 0 else (1.0 if gpu_req == 0 else 0.0)
            
            # Weighted average score
            score = (cpu_score * 0.4 + memory_score * 0.3 + gpu_score * 0.3)
            return True, score
        
        return False, 0.0
    
    def _schedule_timeout_check(self):
        """Schedule the next timeout check"""
        next_check = datetime.now() + timedelta(minutes=1)
        timeout_event = SchedulingEvent(
            timestamp=next_check,
            event_type=SchedulingEventType.TIMEOUT_CHECK,
            priority=10  # Low priority
        )
        self.schedule_event(timeout_event)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        return {
            'current_time': self.current_time,
            'available_agents': len(self.available_agents),
            'busy_agents': len(self.busy_agents),
            'pending_offers': len(self.pending_offers),
            'active_assignments': len(self.job_assignments),
            'event_queue_size': len(self.event_queue)
        }

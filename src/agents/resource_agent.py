import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random
from collections import defaultdict

from .base_agent import BaseAgent
from ..resources.resource import Resource, ResourceStatus
from ..jobs.job import Job, JobStatus
from ..communication.protocol import (
    Message, MessageType, MessagePriority,
    create_resource_offer_message, create_ack_message, create_job_submit_message
)

class ResourceAgent(BaseAgent):
    """Agent that manages a single HPC resource and participates in scheduling"""
    
    def __init__(self, agent_id: str, message_bus, resource: Resource):
        super().__init__(agent_id, message_bus)
        self.resource = resource
        self.pending_offers: Dict[str, Dict] = {}  # job_id -> offer_data
        self.running_jobs: Dict[str, Job] = {}
        self.completed_jobs: List[str] = []
        self.job_history: List[Dict] = []

        # Failure simulation parameters
        self.failure_rate = 0.3  # 30% failure rate
        self.resubmissions = defaultdict(int)  # Job ID to count of resubmissions

        # Scheduling parameters
        self.offer_timeout = 30.0  # seconds
        self.max_concurrent_offers = 5
        self.resource_reservation_time = 300.0  # 5 minutes
        
        # Add resource-specific message handlers
        self.message_handlers.update({
            MessageType.JOB_SUBMIT: self._handle_job_submit,
            MessageType.RESOURCE_REQUEST: self._handle_resource_request,
            MessageType.RESOURCE_RESERVATION: self._handle_resource_reservation,
            MessageType.JOB_CANCEL: self._handle_job_cancel,
            MessageType.NEGOTIATE: self._handle_negotiate
        })
    
    def _run(self):
        """Main resource agent loop"""
        while self.is_running and not self._stop_event.is_set():
            try:
                # Update resource status
                self._update_resource_status()
                
                # Clean up expired offers
                self._cleanup_expired_offers()
                
                # Check running jobs
                self._check_running_jobs()
                
                # Announce resource availability if needed
                self._announce_availability()
                
                # Sleep for a short interval
                self._stop_event.wait(5.0)
                
            except Exception as e:
                self.logger.error(f"Error in resource agent loop: {e}")
                time.sleep(1.0)
    
    def _handle_job_submit(self, message: Message):
        """Handle job submission - evaluate if we can run this job"""
        job_data = message.payload.get("job", {})
        job_id = job_data.get("job_id")
        
        if not job_id or job_id in self.pending_offers:
            return
            
        # Check if we can accommodate this job
        requirements = job_data.get("requirements", {})
        cpu_cores = requirements.get("cpu_cores", 0)
        memory_gb = requirements.get("memory_gb", 0)
        gpu_count = requirements.get("gpu_count", 0)
        storage_gb = requirements.get("storage_gb", 0)
        
        if self.resource.can_accommodate(cpu_cores, memory_gb, gpu_count, storage_gb):
            # Calculate our score/preference for this job
            score = self._calculate_job_score(job_data)
            
            # Create an offer
            offer = self._create_job_offer(job_data, score)
            self.pending_offers[job_id] = {
                "offer": offer,
                "expire_time": datetime.now() + timedelta(seconds=self.offer_timeout),
                "job_data": job_data
            }
            
            # Send offer to the scheduler
            offer_message = create_resource_offer_message(
                self.agent_id,
                "scheduler",  # Always send offers to scheduler
                offer,
                job_id
            )
            self.send_message(offer_message)
            
            self.logger.info(f"Made offer for job {job_id} with score {score}")
    
    def _handle_resource_request(self, message: Message):
        """Handle specific resource requests"""
        job_id = message.payload.get("job_id")
        if job_id and job_id in self.pending_offers:
            # Refresh the offer
            offer_data = self.pending_offers[job_id]
            offer_message = create_resource_offer_message(
                self.agent_id,
                message.sender_id,
                offer_data["offer"],
                job_id
            )
            self.send_message(offer_message)
    
    def _handle_resource_reservation(self, message: Message):
        """Handle resource reservation (job assignment)"""
        job_data = message.payload.get("job", {})
        job_id = job_data.get("job_id")
        
        if not job_id:
            return
            
        # Check if we have a pending offer for this job
        if job_id not in self.pending_offers:
            # Send negative acknowledgment
            ack = create_ack_message(
                self.agent_id,
                message.sender_id,
                message.message_id,
                success=False
            )
            ack.payload["reason"] = "No pending offer for this job"
            self.send_message(ack)
            return
        
        # Try to allocate resources
        offer_data = self.pending_offers[job_id]
        job_data = offer_data["job_data"]
        requirements = job_data.get("requirements", {})
        
        success = self.resource.allocate_resources(
            job_id,
            requirements.get("cpu_cores", 0),
            requirements.get("memory_gb", 0),
            requirements.get("gpu_count", 0),
            requirements.get("storage_gb", 0)
        )
        
        if success:
            # Create job object and start it
            job = self._create_job_from_data(job_data)
            job.status = JobStatus.SCHEDULED
            job.assigned_resource = self.resource.resource_id
            job.start_time = datetime.now()
            
            self.running_jobs[job_id] = job
            
            # Remove from pending offers
            del self.pending_offers[job_id]
            
            # Send positive acknowledgment
            ack = create_ack_message(
                self.agent_id,
                message.sender_id,
                message.message_id,
                success=True
            )
            self.send_message(ack)
            
            self.logger.info(f"Successfully allocated resources for job {job_id}")
            
            # Start job execution simulation
            self._start_job_execution(job)
        else:
            # Send negative acknowledgment
            ack = create_ack_message(
                self.agent_id,
                message.sender_id,
                message.message_id,
                success=False
            )
            ack.payload["reason"] = "Resource allocation failed"
            self.send_message(ack)
    
    def _handle_job_cancel(self, message: Message):
        """Handle job cancellation"""
        job_id = message.payload.get("job_id")
        if job_id in self.running_jobs:
            job = self.running_jobs[job_id]
            self._complete_job(job, JobStatus.CANCELLED)
    
    def _handle_negotiate(self, message: Message):
        """Handle job scheduling negotiations"""
        job_id = message.payload.get("job_id")
        proposal = message.payload.get("proposal", {})
        
        # Simple negotiation: accept if we can still accommodate
        if job_id in self.pending_offers:
            # Check if proposal is acceptable
            acceptable = self._evaluate_proposal(proposal)
            
            response = Message(
                message_id="",
                message_type=MessageType.NEGOTIATE,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                timestamp=datetime.now(),
                priority=MessagePriority.HIGH,
                payload={
                    "job_id": job_id,
                    "response": "accept" if acceptable else "reject",
                    "counter_proposal": self._create_counter_proposal(proposal) if not acceptable else None
                },
                correlation_id=message.message_id
            )
            self.send_message(response)
    
    def _calculate_job_score(self, job_data: Dict) -> float:
        """Calculate how well this job fits our resource"""
        requirements = job_data.get("requirements", {})
        priority = job_data.get("priority", 2)
        
        # Resource utilization efficiency
        cpu_util = requirements.get("cpu_cores", 0) / self.resource.capacity.total_cpu_cores
        mem_util = requirements.get("memory_gb", 0) / self.resource.capacity.total_memory_gb
        gpu_util = (requirements.get("gpu_count", 0) / max(1, self.resource.capacity.total_gpu_count)
                   if self.resource.capacity.total_gpu_count > 0 else 0)
        
        # Prefer jobs that use resources efficiently but not wastefully
        efficiency_score = min(cpu_util + mem_util + gpu_util, 1.0)
        
        # Factor in priority
        priority_score = priority / 4.0
        
        # Factor in current utilization (prefer to balance load)
        current_util = self.resource.utilization_score
        load_balance_score = 1.0 - current_util
        
        # Combine scores
        total_score = (0.4 * efficiency_score + 
                      0.3 * priority_score + 
                      0.3 * load_balance_score)
        
        # Add some randomness to break ties
        total_score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, total_score))
    
    def _create_job_offer(self, job_data: Dict, score: float) -> Dict:
        """Create an offer for a job"""
        return {
            "resource_id": self.resource.resource_id,
            "resource_name": self.resource.name,
            "resource_type": self.resource.resource_type.value,
            "score": score,
            "estimated_start_time": datetime.now().isoformat(),
            "estimated_completion_time": self._estimate_completion_time(job_data).isoformat(),
            "cost_estimate": self._calculate_cost_estimate(job_data),
            "resource_details": {
                "available_cpu": self.resource.capacity.total_cpu_cores - self.resource.utilization.used_cpu_cores,
                "available_memory": self.resource.capacity.total_memory_gb - self.resource.utilization.used_memory_gb,
                "available_gpu": self.resource.capacity.total_gpu_count - self.resource.utilization.used_gpu_count,
                "current_utilization": self.resource.utilization_score
            }
        }
    
    def _estimate_completion_time(self, job_data: Dict) -> datetime:
        """Estimate when the job would complete"""
        requirements = job_data.get("requirements", {})
        estimated_runtime = requirements.get("estimated_runtime_minutes", 60)
        return datetime.now() + timedelta(minutes=estimated_runtime)
    
    def _calculate_cost_estimate(self, job_data: Dict) -> float:
        """Calculate estimated cost for running the job"""
        requirements = job_data.get("requirements", {})
        estimated_runtime = requirements.get("estimated_runtime_minutes", 60) / 60.0  # hours
        return self.resource.cost_per_hour * estimated_runtime
    
    def _start_job_execution(self, job: Job):
        """Start executing a job (simulation)"""
        def execute_job():
            try:
                # Simulate job execution
                execution_time = job.requirements.estimated_runtime_minutes
                execution_time = min(execution_time, 10)  # Cap execution time at 10 seconds for simulation
                if execution_time <= 0:
                    execution_time = random.uniform(1, 10)  # 1-10 seconds for simulation
                
                time.sleep(min(execution_time, 10))  # Cap simulation time at 10 seconds
                
                # Simulate success/failure using configurable failure rate
                success_rate = 1.0 - self.failure_rate
                if random.random() < success_rate:
                    self._complete_job(job, JobStatus.COMPLETED)
                else:
                    # Emulate resubmission if failures occur beyond limit
                    if self.resubmissions[job.job_id] < 3:
                        self.resubmissions[job.job_id] += 1
                        self.logger.info(f"Job {job.job_id} failed; resubmitting ({self.resubmissions[job.job_id]})")
                        job.status = JobStatus.PENDING
                        job.assigned_resource = None
                        job.submit_time = datetime.now()
                        
                        # Send back to job pool for rescheduling
                        self.message_bus.publish(
                            create_job_submit_message("RESOURCE_AGENT", job.to_dict())
                        )
                    else:
                        self.logger.info(f"Job {job.job_id} failed after {self.resubmissions[job.job_id]} attempts")
                        self._complete_job(job, JobStatus.FAILED)
                    
            except Exception as e:
                self.logger.error(f"Error executing job {job.job_id}: {e}")
                self._complete_job(job, JobStatus.FAILED)
        
        # Start job in separate thread
        job_thread = threading.Thread(target=execute_job, daemon=True)
        job_thread.start()
        
        job.status = JobStatus.RUNNING
        self.logger.info(f"Started execution of job {job.job_id}")
    
    def _complete_job(self, job: Job, status: JobStatus):
        """Complete a job and free resources"""
        if job.job_id not in self.running_jobs:
            return
            
        job.status = status
        job.end_time = datetime.now()
        
        # Deallocate resources
        self.resource.deallocate_resources(
            job.job_id,
            job.requirements.cpu_cores,
            job.requirements.memory_gb,
            job.requirements.gpu_count,
            job.requirements.storage_gb
        )
        
        # Move to completed jobs
        del self.running_jobs[job.job_id]
        self.completed_jobs.append(job.job_id)
        
        # Add to history
        self.job_history.append({
            "job_id": job.job_id,
            "status": status.value,
            "runtime_seconds": job.runtime_seconds,
            "completion_time": job.end_time.isoformat()
        })
        
        # Send completion notification
        completion_message = Message(
            message_id="",
            message_type=MessageType.JOB_COMPLETE,
            sender_id=self.agent_id,
            recipient_id="broadcast",
            timestamp=datetime.now(),
            priority=MessagePriority.NORMAL,
            payload={
                "job_id": job.job_id,
                "status": status.value,
                "resource_id": self.resource.resource_id,
                "runtime_seconds": job.runtime_seconds
            }
        )
        self.send_message(completion_message)
        
        self.logger.info(f"Completed job {job.job_id} with status {status.value}")
    
    def _create_job_from_data(self, job_data: Dict) -> Job:
        """Create a Job object from job data dictionary"""
        # This is a simplified conversion - in a real system you'd have proper deserialization
        from ..jobs.job import Job, JobPriority, ResourceRequirement
        
        requirements_data = job_data.get("requirements", {})
        requirements = ResourceRequirement(
            cpu_cores=requirements_data.get("cpu_cores", 1),
            memory_gb=requirements_data.get("memory_gb", 1.0),
            gpu_count=requirements_data.get("gpu_count", 0),
            storage_gb=requirements_data.get("storage_gb", 0),
            estimated_runtime_minutes=requirements_data.get("estimated_runtime_minutes", 60)
        )
        
        return Job(
            job_id=job_data.get("job_id", ""),
            name=job_data.get("name", ""),
            user_id=job_data.get("user_id", ""),
            requirements=requirements,
            priority=JobPriority(job_data.get("priority", 2)),
            command=job_data.get("command", ""),
            working_directory=job_data.get("working_directory", "/tmp"),
            environment_vars=job_data.get("environment_vars", {}),
            input_files=job_data.get("input_files", []),
            output_files=job_data.get("output_files", []),
            dependencies=job_data.get("dependencies", []),
            submit_time=datetime.fromisoformat(job_data.get("submit_time", datetime.now().isoformat()))
        )
    
    def _update_resource_status(self):
        """Update resource status"""
        self.resource.last_heartbeat = datetime.now()
    
    def _cleanup_expired_offers(self):
        """Remove expired offers"""
        current_time = datetime.now()
        expired_offers = [
            job_id for job_id, offer_data in self.pending_offers.items()
            if current_time > offer_data["expire_time"]
        ]
        
        for job_id in expired_offers:
            del self.pending_offers[job_id]
            self.logger.debug(f"Expired offer for job {job_id}")
    
    def _check_running_jobs(self):
        """Check status of running jobs"""
        # In a real system, this would check actual job status
        # For simulation, we rely on the job execution threads
        pass
    
    def _announce_availability(self):
        """Announce resource availability if we have capacity"""
        if (self.resource.status == ResourceStatus.AVAILABLE and 
            len(self.running_jobs) < self.resource.capacity.total_cpu_cores):
            
            # Only announce periodically to avoid spam
            if (datetime.now() - self.last_heartbeat).seconds > 60:
                announcement = Message(
                    message_id="",
                    message_type=MessageType.RESOURCE_ANNOUNCE,
                    sender_id=self.agent_id,
                    recipient_id="broadcast",
                    timestamp=datetime.now(),
                    priority=MessagePriority.LOW,
                    payload={
                        "resource": self.resource.to_dict(),
                        "availability": "high" if self.resource.utilization_score < 0.5 else "medium"
                    }
                )
                self.send_message(announcement)
    
    def _evaluate_proposal(self, proposal: Dict) -> bool:
        """Evaluate a scheduling proposal"""
        # Simple evaluation - accept if we still have capacity
        return self.resource.utilization_score < 0.8
    
    def _create_counter_proposal(self, original_proposal: Dict) -> Dict:
        """Create a counter-proposal"""
        return {
            "suggested_start_time": (datetime.now() + timedelta(minutes=30)).isoformat(),
            "alternative_resources": []
        }
    
    def _get_heartbeat_data(self) -> Dict[str, Any]:
        """Override to include resource-specific data in heartbeat"""
        base_data = super()._get_heartbeat_data()
        base_data.update({
            "resource_id": self.resource.resource_id,
            "resource_utilization": self.resource.utilization_score,
            "running_jobs": len(self.running_jobs),
            "pending_offers": len(self.pending_offers),
            "completed_jobs": len(self.completed_jobs)
        })
        return base_data

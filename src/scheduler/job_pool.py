from typing import List, Dict
from datetime import datetime

from ..communication.protocol import MessageBus, create_job_submit_message
from ..jobs.job import Job, JobStatus

class JobPool:
    """A pool for managing and coordinating submitted jobs"""
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.jobs: Dict[str, Job] = {}
        self.completed_jobs: List[str] = []
    
    def add_job(self, job: Job):
        """Add a job to the pool
        Notify agents about the new job
        """
        self.jobs[job.job_id] = job
        
        # Broadcast job submission
        message = create_job_submit_message(sender_id="job_pool", job_data=job.to_dict())
        self.message_bus.publish(message)
        
    def update_job_status(self, job_id: str, status: JobStatus):
        """Update the status of a job"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = status
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                job.end_time = datetime.now()
                self.completed_jobs.append(job_id)
                del self.jobs[job_id]
                
    def get_pending_jobs(self) -> List[Job]:
        """Get a list of pending jobs"""
        return [job for job in self.jobs.values() if job.status == JobStatus.PENDING]
    
    def get_job(self, job_id: str) -> Job:
        """Get a job by job ID"""
        return self.jobs.get(job_id)
    
    def remove_job(self, job_id: str):
        """Remove a job from the pool"""
        if job_id in self.jobs:
            del self.jobs[job_id]
            
    def get_job_history(self) -> List[Dict]:
        """Get the history of completed jobs"""
        return [self.jobs[job_id].to_dict() for job_id in self.completed_jobs]

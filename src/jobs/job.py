from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import uuid
from datetime import datetime

class JobStatus(Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled" 
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ResourceRequirement:
    """Represents resource requirements for a job"""
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    storage_gb: float = 0
    network_bandwidth_mbps: float = 0
    estimated_runtime_minutes: float = 0
    
@dataclass
class Job:
    """Represents an HPC job with requirements and metadata"""
    job_id: str
    name: str
    user_id: str
    requirements: ResourceRequirement
    priority: JobPriority
    command: str
    working_directory: str
    environment_vars: Dict[str, str]
    input_files: List[str]
    output_files: List[str]
    dependencies: List[str]  # List of job IDs this job depends on
    
    # Metadata
    submit_time: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: JobStatus = JobStatus.PENDING
    assigned_resource: Optional[str] = None  # Resource ID where job is assigned
    
    def __post_init__(self):
        if not self.job_id:
            self.job_id = str(uuid.uuid4())
            
    @property
    def runtime_seconds(self) -> Optional[float]:
        """Calculate actual runtime if job has started and ended"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
        
    def is_ready_to_run(self, completed_jobs: set) -> bool:
        """Check if all dependencies are satisfied"""
        return all(dep_id in completed_jobs for dep_id in self.dependencies)
        
    def to_dict(self) -> Dict:
        """Convert job to dictionary for serialization"""
        return {
            'job_id': self.job_id,
            'name': self.name,
            'user_id': self.user_id,
            'requirements': {
                'cpu_cores': self.requirements.cpu_cores,
                'memory_gb': self.requirements.memory_gb,
                'gpu_count': self.requirements.gpu_count,
                'storage_gb': self.requirements.storage_gb,
                'network_bandwidth_mbps': self.requirements.network_bandwidth_mbps,
                'estimated_runtime_minutes': self.requirements.estimated_runtime_minutes
            },
            'priority': self.priority.value,
            'command': self.command,
            'working_directory': self.working_directory,
            'environment_vars': self.environment_vars,
            'input_files': self.input_files,
            'output_files': self.output_files,
            'dependencies': self.dependencies,
            'submit_time': self.submit_time.isoformat(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status.value,
            'assigned_resource': self.assigned_resource
        }

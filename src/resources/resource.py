from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import uuid
from datetime import datetime

class ResourceType(Enum):
    CPU_CLUSTER = "cpu_cluster"
    GPU_CLUSTER = "gpu_cluster"
    MEMORY_INTENSIVE = "memory_intensive"
    STORAGE_INTENSIVE = "storage_intensive"
    HYBRID = "hybrid"

class ResourceStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

@dataclass
class ResourceCapacity:
    """Represents the total capacity of a resource"""
    total_cpu_cores: int
    total_memory_gb: float
    total_gpu_count: int = 0
    total_storage_gb: float = 0
    max_network_bandwidth_mbps: float = 0

@dataclass
class ResourceUtilization:
    """Represents current utilization of a resource"""
    used_cpu_cores: int = 0
    used_memory_gb: float = 0.0
    used_gpu_count: int = 0
    used_storage_gb: float = 0.0
    used_network_bandwidth_mbps: float = 0.0
    active_jobs: List[str] = field(default_factory=list)
    
    @property
    def cpu_utilization_percent(self) -> float:
        """Calculate CPU utilization percentage"""
        if hasattr(self, '_capacity') and self._capacity.total_cpu_cores > 0:
            return (self.used_cpu_cores / self._capacity.total_cpu_cores) * 100
        return 0.0

@dataclass 
class Resource:
    """Represents an HPC resource (compute node, cluster, etc.)"""
    resource_id: str
    name: str
    resource_type: ResourceType
    capacity: ResourceCapacity
    location: str
    cost_per_hour: float
    
    # Current state
    status: ResourceStatus = ResourceStatus.AVAILABLE
    utilization: ResourceUtilization = field(default_factory=ResourceUtilization)
    last_heartbeat: Optional[datetime] = None
    
    # Metadata
    owner: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.resource_id:
            self.resource_id = str(uuid.uuid4())
        # Link capacity to utilization for percentage calculations
        self.utilization._capacity = self.capacity
            
    def can_accommodate(self, cpu_cores: int, memory_gb: float, 
                       gpu_count: int = 0, storage_gb: float = 0) -> bool:
        """Check if resource can accommodate the given requirements"""
        if self.status != ResourceStatus.AVAILABLE:
            return False
            
        available_cpu = self.capacity.total_cpu_cores - self.utilization.used_cpu_cores
        available_memory = self.capacity.total_memory_gb - self.utilization.used_memory_gb
        available_gpu = self.capacity.total_gpu_count - self.utilization.used_gpu_count
        available_storage = self.capacity.total_storage_gb - self.utilization.used_storage_gb
        
        return (available_cpu >= cpu_cores and
                available_memory >= memory_gb and
                available_gpu >= gpu_count and
                available_storage >= storage_gb)
    
    def allocate_resources(self, job_id: str, cpu_cores: int, memory_gb: float,
                          gpu_count: int = 0, storage_gb: float = 0) -> bool:
        """Allocate resources for a job"""
        if not self.can_accommodate(cpu_cores, memory_gb, gpu_count, storage_gb):
            return False
            
        self.utilization.used_cpu_cores += cpu_cores
        self.utilization.used_memory_gb += memory_gb
        self.utilization.used_gpu_count += gpu_count
        self.utilization.used_storage_gb += storage_gb
        self.utilization.active_jobs.append(job_id)
        
        # Update status if fully utilized
        if (self.utilization.used_cpu_cores >= self.capacity.total_cpu_cores or
            self.utilization.used_memory_gb >= self.capacity.total_memory_gb):
            self.status = ResourceStatus.BUSY
            
        return True
    
    def deallocate_resources(self, job_id: str, cpu_cores: int, memory_gb: float,
                           gpu_count: int = 0, storage_gb: float = 0):
        """Deallocate resources when a job completes"""
        self.utilization.used_cpu_cores = max(0, self.utilization.used_cpu_cores - cpu_cores)
        self.utilization.used_memory_gb = max(0, self.utilization.used_memory_gb - memory_gb)
        self.utilization.used_gpu_count = max(0, self.utilization.used_gpu_count - gpu_count)
        self.utilization.used_storage_gb = max(0, self.utilization.used_storage_gb - storage_gb)
        
        if job_id in self.utilization.active_jobs:
            self.utilization.active_jobs.remove(job_id)
            
        # Update status if resources are available
        if (self.utilization.used_cpu_cores < self.capacity.total_cpu_cores and
            self.utilization.used_memory_gb < self.capacity.total_memory_gb):
            self.status = ResourceStatus.AVAILABLE
    
    @property
    def utilization_score(self) -> float:
        """Calculate overall utilization score (0-1)"""
        cpu_util = self.utilization.used_cpu_cores / max(1, self.capacity.total_cpu_cores)
        mem_util = self.utilization.used_memory_gb / max(1, self.capacity.total_memory_gb)
        gpu_util = (self.utilization.used_gpu_count / max(1, self.capacity.total_gpu_count) 
                   if self.capacity.total_gpu_count > 0 else 0)
        
        # Weighted average (CPU and memory are most important)
        weights = [0.4, 0.4, 0.2] if self.capacity.total_gpu_count > 0 else [0.5, 0.5, 0]
        return weights[0] * cpu_util + weights[1] * mem_util + weights[2] * gpu_util
    
    def to_dict(self) -> Dict:
        """Convert resource to dictionary for serialization"""
        return {
            'resource_id': self.resource_id,
            'name': self.name,
            'resource_type': self.resource_type.value,
            'capacity': {
                'total_cpu_cores': self.capacity.total_cpu_cores,
                'total_memory_gb': self.capacity.total_memory_gb,
                'total_gpu_count': self.capacity.total_gpu_count,
                'total_storage_gb': self.capacity.total_storage_gb,
                'max_network_bandwidth_mbps': self.capacity.max_network_bandwidth_mbps
            },
            'location': self.location,
            'cost_per_hour': self.cost_per_hour,
            'status': self.status.value,
            'utilization': {
                'used_cpu_cores': self.utilization.used_cpu_cores,
                'used_memory_gb': self.utilization.used_memory_gb,
                'used_gpu_count': self.utilization.used_gpu_count,
                'used_storage_gb': self.utilization.used_storage_gb,
                'used_network_bandwidth_mbps': self.utilization.used_network_bandwidth_mbps,
                'active_jobs': self.utilization.active_jobs,
                'utilization_score': self.utilization_score
            },
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'owner': self.owner,
            'tags': self.tags
        }

#!/usr/bin/env python3
"""
HPC Workload Simulator for Fault-Tolerant Distributed Agentic Systems
======================================================================

This module simulates realistic HPC workloads that would require hundreds of nodes,
including scientific computing, machine learning, and data processing tasks.

Realistic HPC Job Types:
1. Computational Fluid Dynamics (CFD) simulations
2. Molecular Dynamics (MD) simulations  
3. Weather/Climate modeling
4. Large-scale Machine Learning training
5. Genomics/Bioinformatics processing
6. Quantum chemistry calculations
7. Seismic data processing
8. Monte Carlo simulations
9. Image/Signal processing pipelines
10. Graph analytics on massive datasets
"""

import random
import time
import math
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

class HPCJobType(Enum):
    """Types of HPC computational jobs"""
    CFD_SIMULATION = "cfd_simulation"
    MOLECULAR_DYNAMICS = "molecular_dynamics" 
    WEATHER_MODELING = "weather_modeling"
    ML_TRAINING = "ml_training"
    GENOMICS = "genomics"
    QUANTUM_CHEMISTRY = "quantum_chemistry"
    SEISMIC_PROCESSING = "seismic_processing"
    MONTE_CARLO = "monte_carlo"
    IMAGE_PROCESSING = "image_processing"
    GRAPH_ANALYTICS = "graph_analytics"

class NodeConfiguration(Enum):
    """Different HPC node configurations"""
    CPU_HEAVY = "cpu_heavy"      # 64-128 cores, 256-512GB RAM
    GPU_ACCELERATED = "gpu_accelerated"  # 4-8 GPUs, 128-256GB RAM
    MEMORY_INTENSIVE = "memory_intensive"  # 16-32 cores, 1-4TB RAM
    STORAGE_OPTIMIZED = "storage_optimized"  # Fast NVMe, 32-64 cores
    NETWORK_OPTIMIZED = "network_optimized"  # InfiniBand, low latency

@dataclass
class HPCNode:
    """Represents a single HPC compute node"""
    node_id: str
    config_type: NodeConfiguration
    cpu_cores: int
    memory_gb: int
    gpu_count: int
    storage_tb: float
    network_bandwidth_gbps: float
    is_active: bool = True
    current_utilization: float = 0.0
    assigned_jobs: List[str] = None
    
    def __post_init__(self):
        if self.assigned_jobs is None:
            self.assigned_jobs = []

@dataclass  
class HPCJob:
    """Represents a computational job requiring HPC resources"""
    job_id: str
    job_type: HPCJobType
    required_nodes: int
    estimated_runtime_hours: float
    cpu_hours_total: float
    memory_gb_peak: float
    storage_gb_required: float
    network_io_intensive: bool
    fault_tolerance_level: str  # "low", "medium", "high"
    checkpoint_interval_minutes: int
    preferred_node_config: NodeConfiguration
    priority: str  # "urgent", "high", "normal", "low"
    
    def get_resource_requirements(self) -> Dict:
        """Get detailed resource requirements for scheduling"""
        return {
            "nodes": self.required_nodes,
            "cpu_hours": self.cpu_hours_total,
            "memory_peak_gb": self.memory_gb_peak,
            "storage_gb": self.storage_gb_required,
            "estimated_runtime": self.estimated_runtime_hours,
            "network_intensive": self.network_io_intensive
        }

class HPCWorkloadGenerator:
    """Generates realistic HPC workloads for testing"""
    
    def __init__(self):
        # Define realistic job characteristics based on actual HPC workloads
        self.job_templates = {
            HPCJobType.CFD_SIMULATION: {
                "nodes_range": (64, 512),
                "runtime_hours": (2, 72),
                "cpu_intensive": True,
                "memory_per_node": (4, 16),  # GB per node
                "storage_gb": (100, 5000),
                "fault_tolerance": "medium",
                "checkpoint_interval": 30,
                "preferred_config": NodeConfiguration.CPU_HEAVY
            },
            HPCJobType.MOLECULAR_DYNAMICS: {
                "nodes_range": (32, 256),
                "runtime_hours": (6, 168),  # Up to 1 week
                "cpu_intensive": True,
                "memory_per_node": (2, 8),
                "storage_gb": (50, 2000),
                "fault_tolerance": "high",
                "checkpoint_interval": 15,
                "preferred_config": NodeConfiguration.CPU_HEAVY
            },
            HPCJobType.WEATHER_MODELING: {
                "nodes_range": (128, 1024),
                "runtime_hours": (1, 24),
                "cpu_intensive": True,
                "memory_per_node": (8, 32),
                "storage_gb": (500, 10000),
                "fault_tolerance": "medium",
                "checkpoint_interval": 20,
                "preferred_config": NodeConfiguration.CPU_HEAVY
            },
            HPCJobType.ML_TRAINING: {
                "nodes_range": (8, 128),
                "runtime_hours": (4, 120),
                "cpu_intensive": False,
                "memory_per_node": (16, 64),
                "storage_gb": (1000, 50000),
                "fault_tolerance": "high",
                "checkpoint_interval": 10,
                "preferred_config": NodeConfiguration.GPU_ACCELERATED
            },
            HPCJobType.GENOMICS: {
                "nodes_range": (16, 256),
                "runtime_hours": (2, 48),
                "cpu_intensive": True,
                "memory_per_node": (16, 128),
                "storage_gb": (1000, 100000),
                "fault_tolerance": "medium",
                "checkpoint_interval": 25,
                "preferred_config": NodeConfiguration.MEMORY_INTENSIVE
            },
            HPCJobType.QUANTUM_CHEMISTRY: {
                "nodes_range": (32, 512),
                "runtime_hours": (8, 200),
                "cpu_intensive": True,
                "memory_per_node": (8, 64),
                "storage_gb": (200, 5000),
                "fault_tolerance": "high",
                "checkpoint_interval": 20,
                "preferred_config": NodeConfiguration.CPU_HEAVY
            },
            HPCJobType.SEISMIC_PROCESSING: {
                "nodes_range": (64, 512),
                "runtime_hours": (3, 72),
                "cpu_intensive": True,
                "memory_per_node": (16, 128),
                "storage_gb": (5000, 200000),
                "fault_tolerance": "medium",
                "checkpoint_interval": 30,
                "preferred_config": NodeConfiguration.STORAGE_OPTIMIZED
            },
            HPCJobType.MONTE_CARLO: {
                "nodes_range": (100, 1000),
                "runtime_hours": (1, 48),
                "cpu_intensive": True,
                "memory_per_node": (1, 4),
                "storage_gb": (10, 500),
                "fault_tolerance": "low",  # Embarrassingly parallel
                "checkpoint_interval": 60,
                "preferred_config": NodeConfiguration.CPU_HEAVY
            },
            HPCJobType.IMAGE_PROCESSING: {
                "nodes_range": (16, 256),
                "runtime_hours": (0.5, 24),
                "cpu_intensive": False,
                "memory_per_node": (8, 32),
                "storage_gb": (1000, 50000),
                "fault_tolerance": "low",
                "checkpoint_interval": 45,
                "preferred_config": NodeConfiguration.GPU_ACCELERATED
            },
            HPCJobType.GRAPH_ANALYTICS: {
                "nodes_range": (32, 512),
                "runtime_hours": (1, 48),
                "cpu_intensive": True,
                "memory_per_node": (32, 256),
                "storage_gb": (500, 20000),
                "fault_tolerance": "medium",
                "checkpoint_interval": 20,
                "preferred_config": NodeConfiguration.MEMORY_INTENSIVE
            }
        }
    
    def generate_job(self, job_type: Optional[HPCJobType] = None) -> HPCJob:
        """Generate a single realistic HPC job"""
        if job_type is None:
            job_type = random.choice(list(HPCJobType))
        
        template = self.job_templates[job_type]
        
        # Generate job parameters based on template
        nodes = random.randint(*template["nodes_range"])
        runtime_hours = random.uniform(*template["runtime_hours"])
        memory_per_node = random.uniform(*template["memory_per_node"])
        storage_gb = random.uniform(*template["storage_gb"])
        
        # Calculate total resource requirements
        total_memory_gb = nodes * memory_per_node
        cpu_hours_total = nodes * runtime_hours * random.uniform(0.7, 1.0)  # Efficiency factor
        
        # Assign priority based on job type and size
        if nodes > 500:
            priority = "urgent"
        elif nodes > 200:
            priority = "high" 
        elif nodes > 50:
            priority = "normal"
        else:
            priority = "low"
        
        job = HPCJob(
            job_id=f"{job_type.value}_{random.randint(1000, 9999)}",
            job_type=job_type,
            required_nodes=nodes,
            estimated_runtime_hours=runtime_hours,
            cpu_hours_total=cpu_hours_total,
            memory_gb_peak=total_memory_gb,
            storage_gb_required=storage_gb,
            network_io_intensive=(job_type in [HPCJobType.SEISMIC_PROCESSING, HPCJobType.WEATHER_MODELING]),
            fault_tolerance_level=template["fault_tolerance"],
            checkpoint_interval_minutes=template["checkpoint_interval"],
            preferred_node_config=template["preferred_config"],
            priority=priority
        )
        
        return job
    
    def generate_workload_mix(self, num_jobs: int) -> List[HPCJob]:
        """Generate a realistic mix of HPC jobs"""
        # Define realistic job distribution based on typical HPC centers
        job_distribution = {
            HPCJobType.CFD_SIMULATION: 0.15,
            HPCJobType.MOLECULAR_DYNAMICS: 0.12,
            HPCJobType.WEATHER_MODELING: 0.08,
            HPCJobType.ML_TRAINING: 0.20,
            HPCJobType.GENOMICS: 0.15,
            HPCJobType.QUANTUM_CHEMISTRY: 0.10,
            HPCJobType.SEISMIC_PROCESSING: 0.05,
            HPCJobType.MONTE_CARLO: 0.05,
            HPCJobType.IMAGE_PROCESSING: 0.06,
            HPCJobType.GRAPH_ANALYTICS: 0.04
        }
        
        jobs = []
        for _ in range(num_jobs):
            # Select job type based on distribution
            rand = random.random()
            cumulative = 0
            selected_type = HPCJobType.CFD_SIMULATION
            
            for job_type, prob in job_distribution.items():
                cumulative += prob
                if rand <= cumulative:
                    selected_type = job_type
                    break
            
            job = self.generate_job(selected_type)
            jobs.append(job)
        
        return jobs

class HPCClusterSimulator:
    """Simulates a realistic HPC cluster environment"""
    
    def __init__(self, cluster_size: int = 512):
        self.cluster_size = cluster_size
        self.nodes = self._create_cluster_nodes()
        self.active_jobs = []
        self.completed_jobs = []
        self.failed_jobs = []
        
    def _create_cluster_nodes(self) -> List[HPCNode]:
        """Create a realistic heterogeneous cluster"""
        nodes = []
        
        # Define node configuration distribution (typical HPC cluster)
        config_distribution = {
            NodeConfiguration.CPU_HEAVY: 0.60,        # Most common
            NodeConfiguration.GPU_ACCELERATED: 0.20,  # Growing segment
            NodeConfiguration.MEMORY_INTENSIVE: 0.10,  # Specialized workloads
            NodeConfiguration.STORAGE_OPTIMIZED: 0.05, # Data-intensive jobs
            NodeConfiguration.NETWORK_OPTIMIZED: 0.05  # Communication-intensive
        }
        
        # Node specifications for each configuration type
        node_specs = {
            NodeConfiguration.CPU_HEAVY: {
                "cpu_cores": (64, 128),
                "memory_gb": (256, 512),
                "gpu_count": (0, 0),
                "storage_tb": (1, 8),
                "network_gbps": (25, 100)
            },
            NodeConfiguration.GPU_ACCELERATED: {
                "cpu_cores": (32, 64),
                "memory_gb": (128, 256),
                "gpu_count": (4, 8),
                "storage_tb": (2, 16),
                "network_gbps": (100, 200)
            },
            NodeConfiguration.MEMORY_INTENSIVE: {
                "cpu_cores": (16, 32),
                "memory_gb": (1024, 4096),
                "gpu_count": (0, 2),
                "storage_tb": (2, 32),
                "network_gbps": (25, 100)
            },
            NodeConfiguration.STORAGE_OPTIMIZED: {
                "cpu_cores": (32, 64),
                "memory_gb": (128, 512),
                "gpu_count": (0, 2),
                "storage_tb": (50, 200),
                "network_gbps": (100, 400)
            },
            NodeConfiguration.NETWORK_OPTIMIZED: {
                "cpu_cores": (32, 64),
                "memory_gb": (128, 256),
                "gpu_count": (0, 4),
                "storage_tb": (1, 8),
                "network_gbps": (200, 800)  # InfiniBand
            }
        }
        
        for i in range(self.cluster_size):
            # Select configuration type based on distribution
            rand = random.random()
            cumulative = 0
            selected_config = NodeConfiguration.CPU_HEAVY
            
            for config, prob in config_distribution.items():
                cumulative += prob
                if rand <= cumulative:
                    selected_config = config
                    break
            
            specs = node_specs[selected_config]
            
            node = HPCNode(
                node_id=f"node_{i:04d}",
                config_type=selected_config,
                cpu_cores=random.randint(*specs["cpu_cores"]),
                memory_gb=random.randint(*specs["memory_gb"]),
                gpu_count=random.randint(*specs["gpu_count"]),
                storage_tb=random.uniform(*specs["storage_tb"]),
                network_bandwidth_gbps=random.uniform(*specs["network_gbps"]),
                current_utilization=random.uniform(0.0, 0.3)  # Some baseline load
            )
            nodes.append(node)
        
        return nodes
    
    def get_cluster_summary(self) -> Dict:
        """Get summary statistics of the cluster"""
        total_cores = sum(node.cpu_cores for node in self.nodes)
        total_memory = sum(node.memory_gb for node in self.nodes)
        total_gpus = sum(node.gpu_count for node in self.nodes)
        total_storage = sum(node.storage_tb for node in self.nodes)
        
        active_nodes = sum(1 for node in self.nodes if node.is_active)
        avg_utilization = sum(node.current_utilization for node in self.nodes if node.is_active) / active_nodes if active_nodes > 0 else 0
        
        config_counts = {}
        for config in NodeConfiguration:
            config_counts[config.value] = sum(1 for node in self.nodes if node.config_type == config)
        
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": active_nodes,
            "total_cpu_cores": total_cores,
            "total_memory_tb": total_memory / 1024,
            "total_gpus": total_gpus,
            "total_storage_pb": total_storage / 1024,  # Petabytes
            "average_utilization": avg_utilization,
            "node_configuration_counts": config_counts
        }
    
    def can_schedule_job(self, job: HPCJob) -> Tuple[bool, List[HPCNode]]:
        """Check if a job can be scheduled and return suitable nodes"""
        # Find available nodes matching job requirements
        suitable_nodes = []
        
        for node in self.nodes:
            if (node.is_active and 
                node.current_utilization < 0.8 and  # Leave some headroom
                (job.preferred_node_config == node.config_type or len(suitable_nodes) < job.required_nodes)):
                suitable_nodes.append(node)
        
        # Check if we have enough resources
        if len(suitable_nodes) >= job.required_nodes:
            return True, suitable_nodes[:job.required_nodes]
        else:
            return False, []
    
    def schedule_job(self, job: HPCJob) -> bool:
        """Attempt to schedule a job on the cluster"""
        can_schedule, assigned_nodes = self.can_schedule_job(job)
        
        if can_schedule:
            # Assign job to nodes
            for node in assigned_nodes:
                node.assigned_jobs.append(job.job_id)
                # Estimate utilization increase (simplified)
                additional_load = min(0.4, 1.0 / len(assigned_nodes))
                node.current_utilization = min(1.0, node.current_utilization + additional_load)
            
            self.active_jobs.append(job)
            return True
        
        return False
    
    def simulate_job_completion(self, job: HPCJob, success_probability: float = 0.95) -> bool:
        """Simulate job completion with potential failures"""
        # Higher chance of failure for longer jobs and more nodes
        failure_probability = (1 - success_probability) * (1 + math.log10(job.required_nodes) / 3)
        failure_probability = min(0.3, failure_probability)  # Cap at 30%
        
        if random.random() < failure_probability:
            self.failed_jobs.append(job)
            return False
        else:
            self.completed_jobs.append(job)
            return True
    
    def inject_cluster_failure(self, failure_type: str, severity: float = 0.1):
        """Inject various types of cluster failures"""
        affected_count = int(len(self.nodes) * severity)
        affected_nodes = random.sample(self.nodes, affected_count)
        
        if failure_type == "node_failure":
            for node in affected_nodes:
                node.is_active = False
        elif failure_type == "network_partition":
            for node in affected_nodes:
                node.network_bandwidth_gbps *= 0.1  # Severe network degradation
        elif failure_type == "resource_exhaustion":
            for node in affected_nodes:
                node.current_utilization = 1.0
        elif failure_type == "storage_failure":
            for node in affected_nodes:
                node.storage_tb *= 0.1
    
    def recover_from_failure(self, failure_type: str):
        """Recover cluster from failures"""
        if failure_type == "node_failure":
            for node in self.nodes:
                if not node.is_active and random.random() < 0.8:  # 80% recovery rate
                    node.is_active = True
        elif failure_type == "network_partition":
            for node in self.nodes:
                if node.network_bandwidth_gbps < 10:  # If severely degraded
                    # Restore to original capacity (simplified)
                    if node.config_type == NodeConfiguration.NETWORK_OPTIMIZED:
                        node.network_bandwidth_gbps = random.uniform(200, 800)
                    else:
                        node.network_bandwidth_gbps = random.uniform(25, 200)

def create_realistic_hpc_experiment_config(cluster_size: int = 512, workload_intensity: str = "medium") -> Dict:
    """Create a realistic HPC experimental configuration"""
    
    workload_configs = {
        "light": {
            "jobs_per_hour": 5,
            "avg_job_size": 64,
            "failure_rate": 0.05
        },
        "medium": {
            "jobs_per_hour": 15,
            "avg_job_size": 128,
            "failure_rate": 0.10
        },
        "heavy": {
            "jobs_per_hour": 30,
            "avg_job_size": 256,
            "failure_rate": 0.15
        }
    }
    
    config = workload_configs[workload_intensity]
    
    return {
        "cluster_size": cluster_size,
        "experiment_duration_hours": 4,  # 4-hour experiment
        "workload_intensity": workload_intensity,
        "jobs_submitted_per_hour": config["jobs_per_hour"],
        "average_job_size_nodes": config["avg_job_size"],
        "baseline_failure_rate": config["failure_rate"],
        "checkpoint_overhead_seconds": random.uniform(30, 180),
        "network_topology": "fat_tree",  # Typical HPC network
        "interconnect": "infiniband",
        "file_system": "lustre_parallel",
        "job_scheduler": "slurm_with_backfill"
    }

def main():
    """Demonstrate HPC workload generation and cluster simulation"""
    print("🖥️  HPC Workload Simulator")
    print("=" * 50)
    
    # Create cluster and workload generator
    cluster = HPCClusterSimulator(cluster_size=512)
    workload_gen = HPCWorkloadGenerator()
    
    # Print cluster summary
    summary = cluster.get_cluster_summary()
    print(f"\n📊 HPC Cluster Summary:")
    print(f"Total Nodes: {summary['total_nodes']}")
    print(f"CPU Cores: {summary['total_cpu_cores']:,}")
    print(f"Memory: {summary['total_memory_tb']:.1f} TB")
    print(f"GPUs: {summary['total_gpus']:,}")
    print(f"Storage: {summary['total_storage_pb']:.2f} PB")
    print(f"Average Utilization: {summary['average_utilization']:.1%}")
    
    print(f"\n🏗️  Node Configuration Distribution:")
    for config, count in summary['node_configuration_counts'].items():
        print(f"  {config}: {count} nodes ({count/summary['total_nodes']:.1%})")
    
    # Generate sample workload
    print(f"\n🔬 Generating Sample HPC Workload...")
    jobs = workload_gen.generate_workload_mix(20)
    
    total_nodes_required = sum(job.required_nodes for job in jobs)
    total_cpu_hours = sum(job.cpu_hours_total for job in jobs)
    
    print(f"Generated {len(jobs)} jobs:")
    print(f"Total Nodes Required: {total_nodes_required:,}")
    print(f"Total CPU Hours: {total_cpu_hours:,.0f}")
    
    # Show job breakdown by type
    job_type_counts = {}
    for job in jobs:
        if job.job_type not in job_type_counts:
            job_type_counts[job.job_type] = 0
        job_type_counts[job.job_type] += 1
    
    print(f"\n📋 Job Type Distribution:")
    for job_type, count in job_type_counts.items():
        print(f"  {job_type.value}: {count} jobs")
    
    # Show some example jobs
    print(f"\n💼 Sample Jobs:")
    for i, job in enumerate(jobs[:5]):
        print(f"  {i+1}. {job.job_type.value}: {job.required_nodes} nodes, "
              f"{job.estimated_runtime_hours:.1f}h, {job.priority} priority")
    
    # Simulate scheduling
    print(f"\n⚙️  Simulating Job Scheduling...")
    scheduled_count = 0
    total_resources_used = 0
    
    for job in jobs:
        if cluster.schedule_job(job):
            scheduled_count += 1
            total_resources_used += job.required_nodes
    
    print(f"Successfully scheduled: {scheduled_count}/{len(jobs)} jobs")
    print(f"Cluster utilization: {total_resources_used/cluster.cluster_size:.1%}")
    
    # Simulate failure and recovery
    print(f"\n🔥 Simulating Cluster Failures...")
    cluster.inject_cluster_failure("node_failure", severity=0.1)
    print(f"Injected node failures affecting 10% of cluster")
    
    # Check cluster health
    active_nodes = sum(1 for node in cluster.nodes if node.is_active)
    print(f"Active nodes after failure: {active_nodes}/{cluster.cluster_size}")
    
    # Recovery
    cluster.recover_from_failure("node_failure")
    active_nodes_after = sum(1 for node in cluster.nodes if node.is_active)
    print(f"Active nodes after recovery: {active_nodes_after}/{cluster.cluster_size}")
    
    # Generate experimental configuration
    hpc_config = create_realistic_hpc_experiment_config(
        cluster_size=512,
        workload_intensity="medium"
    )
    
    print(f"\n🧪 Realistic HPC Experiment Configuration:")
    for key, value in hpc_config.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()

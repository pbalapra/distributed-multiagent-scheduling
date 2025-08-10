#!/usr/bin/env python3
"""
Realistic HPC Resource and Job Generator
========================================

Generates realistic HPC cluster configurations and scientific workloads
for multi-agent consensus-based HPC federation simulations.

Features:
- Realistic HPC cluster specifications based on real supercomputers
- Scientific workload types with appropriate resource requirements
- JSON output for integration with simulation systems
- Configurable cluster counts and job generation
"""

import json
import random
import argparse
from typing import Dict, List, Any
from dataclasses import dataclass, asdict


@dataclass
class HPCCluster:
    name: str
    description: str
    nodes: int
    cpu_per_node: int
    memory_per_node: int  # GB
    gpu_per_node: int
    interconnect: str
    specialization: str
    total_cpu_cores: int = 0
    total_memory_gb: int = 0
    total_gpus: int = 0
    
    def __post_init__(self):
        self.total_cpu_cores = self.nodes * self.cpu_per_node
        self.total_memory_gb = self.nodes * self.memory_per_node
        self.total_gpus = self.nodes * self.gpu_per_node


@dataclass
class ScientificJob:
    job_id: str
    workload_type: str
    cpu_cores: int
    memory_gb: int
    gpu_count: int
    estimated_runtime: float  # seconds
    priority: float
    description: str
    scientific_domain: str


class HPCResourceGenerator:
    """Generate realistic HPC cluster configurations"""
    
    def __init__(self):
        # Based on real supercomputer architectures
        self.cluster_templates = [
            {
                "name_prefix": "SUMMIT",
                "description": "GPU-Heavy Cluster (AI/ML workloads)",
                "cpu_per_node": 44,  # IBM Power9
                "memory_per_node": 512,  # GB
                "gpu_per_node": 6,  # NVIDIA V100
                "interconnect": "EDR InfiniBand",
                "specialization": "Deep Learning, AI Training, GPU Computing"
            },
            {
                "name_prefix": "FRONTIER", 
                "description": "CPU-Heavy Cluster (Traditional HPC)",
                "cpu_per_node": 64,  # AMD EPYC
                "memory_per_node": 256,  # GB
                "gpu_per_node": 0,
                "interconnect": "HDR InfiniBand",
                "specialization": "CFD, Molecular Dynamics, Weather Simulation"
            },
            {
                "name_prefix": "HYBRID",
                "description": "Balanced CPU/GPU Cluster",
                "cpu_per_node": 56,  # Intel Xeon
                "memory_per_node": 384,  # GB
                "gpu_per_node": 2,  # NVIDIA A100
                "interconnect": "HDR InfiniBand", 
                "specialization": "Climate Modeling, Bioinformatics"
            },
            {
                "name_prefix": "MEMORY",
                "description": "High-Memory Cluster (Large datasets)",
                "cpu_per_node": 48,  # Intel Xeon
                "memory_per_node": 1024,  # GB (1TB per node)
                "gpu_per_node": 1,  # NVIDIA H100
                "interconnect": "HDR InfiniBand",
                "specialization": "Genomics, Large-scale Simulations"
            },
            {
                "name_prefix": "QUANTUM",
                "description": "Quantum-Classical Hybrid Cluster",
                "cpu_per_node": 32,  # Intel Xeon
                "memory_per_node": 768,  # GB
                "gpu_per_node": 4,  # NVIDIA A100
                "interconnect": "HDR InfiniBand + Quantum",
                "specialization": "Quantum Computing, Optimization"
            }
        ]
    
    def generate_clusters(self, num_clusters: int, nodes_per_cluster: int = 100) -> List[HPCCluster]:
        """Generate realistic HPC cluster configurations"""
        clusters = []
        
        for i in range(num_clusters):
            # Select template (round-robin to ensure diversity)
            template = self.cluster_templates[i % len(self.cluster_templates)]
            
            # Add some variability to node count
            node_count = random.randint(
                int(nodes_per_cluster * 0.8), 
                int(nodes_per_cluster * 1.2)
            )
            
            cluster = HPCCluster(
                name=f"{template['name_prefix']}_{i+1:02d}",
                description=template["description"],
                nodes=node_count,
                cpu_per_node=template["cpu_per_node"],
                memory_per_node=template["memory_per_node"],
                gpu_per_node=template["gpu_per_node"],
                interconnect=template["interconnect"],
                specialization=template["specialization"]
            )
            
            clusters.append(cluster)
            
        return clusters


class ScientificJobGenerator:
    """Generate realistic scientific HPC workloads"""
    
    def __init__(self):
        self.workload_types = [
            {
                "name": "Deep Learning Training",
                "cpu_factor": 0.3,     # GPU-bound, moderate CPU
                "memory_factor": 0.4,  # Moderate memory per GPU
                "gpu_factor": 0.8,     # GPU-intensive
                "runtime_range": (1800, 10800),  # 30 minutes to 3 hours
                "priority_range": (0.7, 1.0),
                "domain": "Artificial Intelligence",
                "description": "Large-scale neural network training",
                "weight": 0.2
            },
            {
                "name": "Molecular Dynamics Simulation",
                "cpu_factor": 0.6,     # CPU-intensive
                "memory_factor": 0.3,  # Moderate memory
                "gpu_factor": 0.1,     # Occasional GPU acceleration
                "runtime_range": (3600, 18000),  # 1-5 hours
                "priority_range": (0.5, 0.8),
                "domain": "Computational Chemistry",
                "description": "Protein folding and drug discovery",
                "weight": 0.25
            },
            {
                "name": "Climate Simulation",
                "cpu_factor": 0.7,     # Very CPU-intensive
                "memory_factor": 0.5,  # Memory-intensive
                "gpu_factor": 0.2,     # Some GPU acceleration
                "runtime_range": (7200, 36000),  # 2-10 hours
                "priority_range": (0.6, 0.9),
                "domain": "Earth Sciences",
                "description": "Global climate modeling and prediction",
                "weight": 0.2
            },
            {
                "name": "Genomics Analysis",
                "cpu_factor": 0.4,     # Moderate CPU
                "memory_factor": 0.8,  # Very memory-intensive
                "gpu_factor": 0.05,    # Rarely uses GPU
                "runtime_range": (2700, 14400),  # 45 minutes to 4 hours
                "priority_range": (0.4, 0.7),
                "domain": "Bioinformatics",
                "description": "Whole genome sequencing analysis",
                "weight": 0.15
            },
            {
                "name": "Computational Fluid Dynamics",
                "cpu_factor": 0.8,     # Very CPU-intensive
                "memory_factor": 0.4,  # Moderate memory
                "gpu_factor": 0.1,     # Some GPU acceleration
                "runtime_range": (5400, 21600),  # 1.5-6 hours
                "priority_range": (0.5, 0.8),
                "domain": "Engineering",
                "description": "Aerodynamics and fluid flow simulation",
                "weight": 0.2
            }
        ]
    
    def generate_jobs(self, num_jobs: int, max_cpu: int, max_memory: int, max_gpu: int) -> List[ScientificJob]:
        """Generate realistic scientific job workloads"""
        jobs = []
        
        for i in range(num_jobs):
            # Select workload type based on weights
            workload = random.choices(
                self.workload_types, 
                weights=[w["weight"] for w in self.workload_types]
            )[0]
            
            # Calculate resources based on workload characteristics
            cpu_cores = max(16, int(max_cpu * workload["cpu_factor"] * random.uniform(0.05, 0.8)))
            memory_gb = max(32, int(max_memory * workload["memory_factor"] * random.uniform(0.05, 0.6)))
            
            # GPU allocation based on workload type
            if workload["gpu_factor"] > 0.5:  # GPU-intensive workloads
                gpu_count = random.randint(max(1, int(max_gpu * 0.3)), int(max_gpu * 0.9))
            elif workload["gpu_factor"] > 0.15:  # Moderate GPU usage
                gpu_count = random.choices([0, 1, 2, 4], weights=[0.3, 0.4, 0.2, 0.1])[0]
            else:  # CPU-only workloads
                gpu_count = random.choices([0, 1], weights=[0.85, 0.15])[0]
            
            # Ensure GPU count doesn't exceed available
            gpu_count = min(gpu_count, max_gpu)
            
            # Realistic runtime
            runtime_min, runtime_max = workload["runtime_range"]
            estimated_runtime = random.uniform(runtime_min, runtime_max)
            
            # Priority based on workload type
            priority_min, priority_max = workload["priority_range"]
            priority = random.uniform(priority_min, priority_max)
            
            job = ScientificJob(
                job_id=f"SCI_{i+1:03d}",
                workload_type=workload["name"],
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                gpu_count=gpu_count,
                estimated_runtime=estimated_runtime,
                priority=priority,
                description=workload["description"],
                scientific_domain=workload["domain"]
            )
            
            jobs.append(job)
        
        return jobs


def print_cluster_summary(clusters: List[HPCCluster]):
    """Print summary of generated clusters"""
    print("🌐 GENERATED HPC FEDERATION CLUSTERS")
    print("=" * 80)
    
    total_nodes = 0
    total_cpu = 0
    total_memory = 0
    total_gpu = 0
    
    for cluster in clusters:
        print(f"🖥️  {cluster.name}: {cluster.description}")
        print(f"    Nodes: {cluster.nodes:,}")
        print(f"    Total CPU Cores: {cluster.total_cpu_cores:,}")
        print(f"    Total Memory: {cluster.total_memory_gb:,} GB")
        print(f"    Total GPUs: {cluster.total_gpus:,}")
        print(f"    Interconnect: {cluster.interconnect}")
        print(f"    Specialization: {cluster.specialization}")
        print()
        
        total_nodes += cluster.nodes
        total_cpu += cluster.total_cpu_cores
        total_memory += cluster.total_memory_gb
        total_gpu += cluster.total_gpus
    
    print(f"🔗 FEDERATION TOTALS:")
    print(f"    Total Nodes: {total_nodes:,}")
    print(f"    Total CPU Cores: {total_cpu:,}")
    print(f"    Total Memory: {total_memory:,} GB ({total_memory/1024:.1f} TB)")
    print(f"    Total GPUs: {total_gpu:,}")
    print("=" * 80)


def print_job_summary(jobs: List[ScientificJob]):
    """Print summary of generated jobs"""
    print("\n🔬 GENERATED SCIENTIFIC WORKLOADS")
    print("=" * 80)
    
    # Count by workload type
    workload_counts = {}
    total_cpu = 0
    total_memory = 0
    total_gpu = 0
    
    for job in jobs:
        workload_counts[job.workload_type] = workload_counts.get(job.workload_type, 0) + 1
        total_cpu += job.cpu_cores
        total_memory += job.memory_gb
        total_gpu += job.gpu_count
    
    print(f"📊 WORKLOAD DISTRIBUTION:")
    for workload_type, count in workload_counts.items():
        percentage = (count / len(jobs)) * 100
        print(f"    {workload_type}: {count} jobs ({percentage:.1f}%)")
    
    print(f"\n💼 RESOURCE REQUIREMENTS:")
    print(f"    Total CPU: {total_cpu:,} cores")
    print(f"    Total Memory: {total_memory:,} GB ({total_memory/1024:.1f} TB)")
    print(f"    Total GPUs: {total_gpu:,}")
    print(f"    Average CPU per job: {total_cpu/len(jobs):.0f} cores")
    print(f"    Average Memory per job: {total_memory/len(jobs):.0f} GB")
    print(f"    Average GPUs per job: {total_gpu/len(jobs):.1f}")
    
    print("\n🎯 SAMPLE JOBS:")
    for i, job in enumerate(jobs[:5]):
        print(f"    {job.job_id}: {job.workload_type}")
        print(f"        Resources: {job.cpu_cores:,} CPU, {job.memory_gb:,}GB RAM, {job.gpu_count} GPU")
        print(f"        Runtime: {job.estimated_runtime/3600:.1f}h, Priority: {job.priority:.2f}")
        print(f"        Domain: {job.scientific_domain}")
    
    if len(jobs) > 5:
        print(f"    ... and {len(jobs) - 5} more jobs")
    
    print("=" * 80)


def save_to_json(clusters: List[HPCCluster], jobs: List[ScientificJob], output_prefix: str = "hpc_federation"):
    """Save generated data to JSON files"""
    
    # Convert to dictionaries for JSON serialization
    cluster_data = {
        "metadata": {
            "generator": "HPC Resource Generator",
            "timestamp": "2025-01-16",
            "total_clusters": len(clusters),
            "federation_capacity": {
                "total_nodes": sum(c.nodes for c in clusters),
                "total_cpu_cores": sum(c.total_cpu_cores for c in clusters),
                "total_memory_gb": sum(c.total_memory_gb for c in clusters),
                "total_gpus": sum(c.total_gpus for c in clusters)
            }
        },
        "clusters": [asdict(cluster) for cluster in clusters]
    }
    
    job_data = {
        "metadata": {
            "generator": "Scientific Job Generator",
            "timestamp": "2025-01-16", 
            "total_jobs": len(jobs),
            "workload_summary": {
                "total_cpu_required": sum(j.cpu_cores for j in jobs),
                "total_memory_required": sum(j.memory_gb for j in jobs),
                "total_gpus_required": sum(j.gpu_count for j in jobs)
            }
        },
        "jobs": [asdict(job) for job in jobs]
    }
    
    # Save to files
    clusters_file = f"{output_prefix}_clusters.json"
    jobs_file = f"{output_prefix}_jobs.json"
    
    with open(clusters_file, 'w') as f:
        json.dump(cluster_data, f, indent=2)
    
    with open(jobs_file, 'w') as f:
        json.dump(job_data, f, indent=2)
    
    print(f"\n💾 SAVED TO FILES:")
    print(f"    Clusters: {clusters_file}")
    print(f"    Jobs: {jobs_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate realistic HPC clusters and scientific jobs")
    parser.add_argument("--clusters", type=int, default=4, help="Number of HPC clusters to generate")
    parser.add_argument("--nodes_per_cluster", type=int, default=100, help="Average nodes per cluster")
    parser.add_argument("--jobs", type=int, default=20, help="Number of scientific jobs to generate")
    parser.add_argument("--output", type=str, default="hpc_federation", help="Output file prefix")
    
    args = parser.parse_args()
    
    print("🔬 REALISTIC HPC RESOURCE & JOB GENERATOR")
    print("=" * 80)
    print(f"Generating {args.clusters} HPC clusters with ~{args.nodes_per_cluster} nodes each")
    print(f"Generating {args.jobs} realistic scientific workloads")
    print("=" * 80)
    
    # Generate clusters
    generator = HPCResourceGenerator()
    clusters = generator.generate_clusters(args.clusters, args.nodes_per_cluster)
    print_cluster_summary(clusters)
    
    # Calculate federation capacity for job generation
    max_cpu = max(cluster.total_cpu_cores for cluster in clusters)
    max_memory = max(cluster.total_memory_gb for cluster in clusters)
    max_gpu = max(cluster.total_gpus for cluster in clusters)
    
    # Generate scientific jobs
    job_generator = ScientificJobGenerator()
    jobs = job_generator.generate_jobs(args.jobs, max_cpu, max_memory, max_gpu)
    print_job_summary(jobs)
    
    # Save to JSON files
    save_to_json(clusters, jobs, args.output)
    
    print("\n✅ HPC RESOURCE AND JOB GENERATION COMPLETE")
    print("🎯 Ready for decentralized consensus-based HPC federation simulation!")


if __name__ == "__main__":
    main()

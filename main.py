import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from src.communication.protocol import MessageBus
from src.scheduler.job_pool import JobPool
from src.scheduler.scheduler import Scheduler
from src.agents.resource_agent import ResourceAgent
from src.resources.resource import Resource, ResourceCapacity, ResourceType

# Create a message bus for agent communication
message_bus = MessageBus()

# Create a job pool
job_pool = JobPool(message_bus)

# Create a scheduler
scheduler = Scheduler(message_bus, job_pool)

# Create some sample resources
resources = [
    Resource("resource-1", "CPU-Cluster-1", ResourceType.CPU_CLUSTER,
             ResourceCapacity(total_cpu_cores=32, total_memory_gb=64), "location-1", 10.0),
    Resource("resource-2", "GPU-Cluster-1", ResourceType.GPU_CLUSTER,
             ResourceCapacity(total_cpu_cores=16, total_memory_gb=32, total_gpu_count=4), "location-2", 20.0),
    Resource("resource-3", "Hybrid-Cluster-1", ResourceType.HYBRID,
             ResourceCapacity(total_cpu_cores=24, total_memory_gb=48, total_gpu_count=2), "location-3", 15.0)
]

# Create resource agents
agents = [ResourceAgent(f"agent-{i+1}", message_bus, resources[i]) for i in range(len(resources))]

# Start all components
scheduler.start()
for agent in agents:
    agent.start()

# Example: Add a job to the job pool
#from src.jobs.job import Job, JobPriority, ResourceRequirement
#job = Job(
#    job_id="",
#    name="Example Job",
#    user_id="user1",
#    requirements=ResourceRequirement(cpu_cores=4, memory_gb=8),
#    priority=JobPriority.MEDIUM,
#    command="echo 'Hello World'",
#    working_directory="/tmp",
#    environment_vars={},
#    input_files=[],
#    output_files=[],
#    dependencies=[],
#    submit_time=datetime.now()
#)
#
#job_pool.add_job(job)

# Simulation running...
try:
    while True:
        pass  # Keep the main script running
except KeyboardInterrupt:
    # Stop all components on interruption
    scheduler.stop()
    for agent in agents:
        agent.stop()

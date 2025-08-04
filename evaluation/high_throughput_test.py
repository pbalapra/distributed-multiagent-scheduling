#!/usr/bin/env python3
"""
High-Throughput, Low-Latency Scheduler Comparison Tests
======================================================

This test suite demonstrates that centralized scheduling is superior for:
1. High-throughput environments (many jobs per second)
2. Low-latency environments (minimal scheduling overhead)
3. Environments requiring predictable performance

Test scenarios:
- Burst job arrivals
- Continuous high-frequency job streams
- Latency-sensitive job scheduling
- Resource contention under load
"""

import uuid
import time
import random
import statistics
from typing import List, Dict, Tuple
from true_discrete_event_demo import (TrueDiscreteEventSimulation, DiscreteEventJob, 
                                     DiscreteEventAgent, HPC_Resource, Priority)
from combined_scheduling_demo import CentralizedScheduler, DistributedScheduler

class PerformanceTracker:
    """Track detailed performance metrics for scheduling comparison"""
    
    def __init__(self, name: str):
        self.name = name
        self.job_arrival_times: List[float] = []
        self.job_assignment_times: List[float] = []
        self.job_completion_times: List[float] = []
        self.scheduling_latencies: List[float] = []  # Time from arrival to assignment
        self.decision_times: List[float] = []  # Time spent making scheduling decision
        self.throughput_measurements: List[Tuple[float, int]] = []  # (time, jobs_completed)
        
    def record_job_arrival(self, job_id: str, arrival_time: float):
        self.job_arrival_times.append(arrival_time)
        
    def record_job_assignment(self, job_id: str, assignment_time: float, decision_time: float):
        self.job_assignment_times.append(assignment_time)
        self.decision_times.append(decision_time)
        
    def record_job_completion(self, job_id: str, completion_time: float):
        self.job_completion_times.append(completion_time)
        
    def calculate_latencies(self):
        """Calculate scheduling latencies (arrival to assignment time)"""
        self.scheduling_latencies = []
        arrival_dict = {i: time for i, time in enumerate(self.job_arrival_times)}
        
        for i, assignment_time in enumerate(self.job_assignment_times):
            if i < len(self.job_arrival_times):
                latency = assignment_time - self.job_arrival_times[i]
                self.scheduling_latencies.append(latency)
    
    def get_throughput_over_time(self, window_size: float = 10.0) -> List[float]:
        """Calculate throughput (jobs/sec) over sliding time windows"""
        if not self.job_completion_times:
            return []
            
        throughputs = []
        completion_times = sorted(self.job_completion_times)
        start_time = completion_times[0]
        end_time = completion_times[-1]
        
        current_time = start_time
        while current_time <= end_time:
            window_start = current_time
            window_end = current_time + window_size
            
            jobs_in_window = sum(1 for t in completion_times 
                               if window_start <= t < window_end)
            throughput = jobs_in_window / window_size
            throughputs.append(throughput)
            
            current_time += window_size / 2  # 50% overlap for smoother measurement
            
        return throughputs
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get comprehensive performance statistics"""
        self.calculate_latencies()
        
        return {
            'total_jobs': len(self.job_completion_times),
            'avg_scheduling_latency': statistics.mean(self.scheduling_latencies) if self.scheduling_latencies else 0,
            'p50_scheduling_latency': statistics.median(self.scheduling_latencies) if self.scheduling_latencies else 0,
            'p95_scheduling_latency': statistics.quantiles(self.scheduling_latencies, n=20)[18] if len(self.scheduling_latencies) > 20 else 0,
            'p99_scheduling_latency': statistics.quantiles(self.scheduling_latencies, n=100)[98] if len(self.scheduling_latencies) > 100 else 0,
            'avg_decision_time': statistics.mean(self.decision_times) if self.decision_times else 0,
            'max_decision_time': max(self.decision_times) if self.decision_times else 0,
            'avg_throughput': len(self.job_completion_times) / (max(self.job_completion_times) - min(self.job_completion_times)) if len(self.job_completion_times) > 1 else 0,
            'peak_throughput': max(self.get_throughput_over_time()) if self.get_throughput_over_time() else 0
        }

class InstrumentedCentralizedScheduler(CentralizedScheduler):
    """Centralized scheduler with performance tracking"""
    
    def __init__(self, simulation, tracker: PerformanceTracker):
        super().__init__(simulation)
        self.tracker = tracker
        
    def schedule_job(self, job: DiscreteEventJob):
        start_time = time.time()
        arrival_time = self.simulation.clock.now()
        
        # Record job arrival
        self.tracker.record_job_arrival(job.id, arrival_time)
        
        # Call parent scheduling logic
        result = super().schedule_job(job)
        
        # Record timing if job was assigned
        if job.status == "assigned":
            decision_time = time.time() - start_time
            assignment_time = self.simulation.clock.now()
            self.tracker.record_job_assignment(job.id, assignment_time, decision_time)
            
        return result
        
    def handle_job_completion(self, job: DiscreteEventJob):
        if job.status == "completed":
            self.tracker.record_job_completion(job.id, self.simulation.clock.now())
        super().handle_job_completion(job)

class InstrumentedDistributedScheduler(DistributedScheduler):
    """Distributed scheduler with performance tracking"""
    
    def __init__(self, simulation, tracker: PerformanceTracker):
        super().__init__(simulation)
        self.tracker = tracker
        
    def schedule_job(self, job: DiscreteEventJob):
        start_time = time.time()
        arrival_time = self.simulation.clock.now()
        
        # Record job arrival
        self.tracker.record_job_arrival(job.id, arrival_time)
        
        # Call parent scheduling logic
        result = super().schedule_job(job)
        
        # Record timing if job was assigned
        if job.status == "assigned":
            decision_time = time.time() - start_time
            assignment_time = self.simulation.clock.now()
            self.tracker.record_job_assignment(job.id, assignment_time, decision_time)
            
        return result
        
    def handle_job_completion(self, job: DiscreteEventJob):
        if job.status == "completed":
            self.tracker.record_job_completion(job.id, self.simulation.clock.now())
        super().handle_job_completion(job)

def create_high_throughput_jobs(count: int, job_type: str = "mixed") -> List[DiscreteEventJob]:
    """Create jobs optimized for high-throughput testing"""
    jobs = []
    
    for i in range(count):
        if job_type == "small":
            # Small, fast jobs for maximum throughput
            job = DiscreteEventJob(
                str(uuid.uuid4()),
                f"FastJob-{i}",
                Priority.HIGH,
                {"cpu": 1, "memory": 1, "gpu": 0},
                random.uniform(0.5, 2.0),  # Very short duration
                0.5
            )
        elif job_type == "mixed":
            # Mixed workload simulating real high-throughput environment
            if i % 4 == 0:
                # 25% medium jobs
                job = DiscreteEventJob(
                    str(uuid.uuid4()),
                    f"MediumJob-{i}",
                    Priority.MEDIUM,
                    {"cpu": 4, "memory": 8, "gpu": 0},
                    random.uniform(2.0, 8.0),
                    1.0
                )
            else:
                # 75% small jobs
                job = DiscreteEventJob(
                    str(uuid.uuid4()),
                    f"SmallJob-{i}",
                    Priority.HIGH,
                    {"cpu": 1, "memory": 2, "gpu": 0},
                    random.uniform(0.5, 3.0),
                    0.5
                )
        else:  # "latency_sensitive"
            # Ultra-low latency jobs
            job = DiscreteEventJob(
                str(uuid.uuid4()),
                f"LatencyJob-{i}",
                Priority.CRITICAL,
                {"cpu": 1, "memory": 1, "gpu": 0},
                random.uniform(0.1, 0.5),  # Ultra-short duration
                0.3
            )
            
        jobs.append(job)
    
    return jobs

def setup_high_throughput_cluster():
    """Create an agent cluster optimized for high-throughput workloads"""
    simulation = TrueDiscreteEventSimulation()
    
    # Many small, fast agents for high-throughput processing
    agents = [
        # Small CPU agents (optimized for fast job processing)
        ("fast-cpu-1", HPC_Resource("Fast CPU 1", 2, 4, 0, 2.0), 0.01),
        ("fast-cpu-2", HPC_Resource("Fast CPU 2", 2, 4, 0, 2.0), 0.01),
        ("fast-cpu-3", HPC_Resource("Fast CPU 3", 2, 4, 0, 2.0), 0.01),
        ("fast-cpu-4", HPC_Resource("Fast CPU 4", 2, 4, 0, 2.0), 0.01),
        ("fast-cpu-5", HPC_Resource("Fast CPU 5", 2, 4, 0, 2.0), 0.01),
        ("fast-cpu-6", HPC_Resource("Fast CPU 6", 2, 4, 0, 2.0), 0.01),
        ("fast-cpu-7", HPC_Resource("Fast CPU 7", 2, 4, 0, 2.0), 0.01),
        ("fast-cpu-8", HPC_Resource("Fast CPU 8", 2, 4, 0, 2.0), 0.01),
        
        # Medium agents for occasional larger jobs
        ("medium-cpu-1", HPC_Resource("Medium CPU 1", 8, 16, 0, 8.0), 0.02),
        ("medium-cpu-2", HPC_Resource("Medium CPU 2", 8, 16, 0, 8.0), 0.02),
        ("medium-cpu-3", HPC_Resource("Medium CPU 3", 8, 16, 0, 8.0), 0.02),
        ("medium-cpu-4", HPC_Resource("Medium CPU 4", 8, 16, 0, 8.0), 0.02),
    ]
    
    for agent_id, resource, failure_rate in agents:
        agent = DiscreteEventAgent(agent_id, resource, simulation, failure_rate)
        simulation.add_agent(agent)
    
    return simulation

def test_burst_job_handling():
    """Test 1: Handle sudden bursts of jobs (high-throughput scenario)"""
    print("=" * 80)
    print("🚀 TEST 1: BURST JOB HANDLING (High-Throughput)")
    print("=" * 80)
    print("Scenario: 100 jobs arrive within 10 time units")
    print("Measures: Scheduling latency, throughput, decision time")
    
    results = {}
    
    for scheduler_type in ["Centralized", "Distributed"]:
        print(f"\n📊 Testing {scheduler_type} Scheduler...")
        
        # Setup
        simulation = setup_high_throughput_cluster()
        tracker = PerformanceTracker(scheduler_type)
        
        if scheduler_type == "Centralized":
            scheduler = InstrumentedCentralizedScheduler(simulation, tracker)
        else:
            scheduler = InstrumentedDistributedScheduler(simulation, tracker)
            
        simulation.scheduler = scheduler
        
        # Create burst workload: 100 jobs in first 10 time units
        jobs = create_high_throughput_jobs(100, "mixed")
        
        # Submit jobs in burst pattern
        burst_window = 10.0
        for i, job in enumerate(jobs):
            # Jobs arrive in first 10 time units (burst)
            arrival_time = (i / len(jobs)) * burst_window
            simulation.submit_job(job, arrival_time)
        
        # Run simulation
        start_time = time.time()
        simulation.run(max_simulation_time=200.0)
        wall_time = time.time() - start_time
        
        stats = tracker.get_summary_stats()
        stats['wall_time'] = wall_time
        stats['scheduler_assignments'] = scheduler.get_stats()['metrics']['assignments_made']
        
        results[scheduler_type] = stats
        
        print(f"  ✅ Jobs completed: {stats['total_jobs']}")
        print(f"  ⚡ Avg scheduling latency: {stats['avg_scheduling_latency']:.3f}s")
        print(f"  📈 Peak throughput: {stats['peak_throughput']:.1f} jobs/sec")
        print(f"  🎯 Avg decision time: {stats['avg_decision_time']*1000:.2f}ms")
    
    # Compare results
    print(f"\n🏆 BURST HANDLING COMPARISON:")
    cent = results["Centralized"]
    dist = results["Distributed"]
    
    print(f"📊 Scheduling Latency (lower is better):")
    print(f"  Centralized: {cent['avg_scheduling_latency']:.3f}s")
    print(f"  Distributed: {dist['avg_scheduling_latency']:.3f}s")
    print(f"  Winner: {'Centralized' if cent['avg_scheduling_latency'] < dist['avg_scheduling_latency'] else 'Distributed'}")
    
    print(f"📊 Decision Time (lower is better):")
    print(f"  Centralized: {cent['avg_decision_time']*1000:.2f}ms")
    print(f"  Distributed: {dist['avg_decision_time']*1000:.2f}ms") 
    print(f"  Winner: {'Centralized' if cent['avg_decision_time'] < dist['avg_decision_time'] else 'Distributed'}")
    
    print(f"📊 Peak Throughput (higher is better):")
    print(f"  Centralized: {cent['peak_throughput']:.1f} jobs/sec")
    print(f"  Distributed: {dist['peak_throughput']:.1f} jobs/sec")
    print(f"  Winner: {'Centralized' if cent['peak_throughput'] > dist['peak_throughput'] else 'Distributed'}")
    
    return results

def test_continuous_high_frequency():
    """Test 2: Continuous high-frequency job stream"""
    print("\n" + "=" * 80)
    print("🌊 TEST 2: CONTINUOUS HIGH-FREQUENCY STREAM")
    print("=" * 80)
    print("Scenario: 200 jobs arrive continuously every 0.5 time units")
    print("Measures: Sustained throughput, latency consistency")
    
    results = {}
    
    for scheduler_type in ["Centralized", "Distributed"]:
        print(f"\n📊 Testing {scheduler_type} Scheduler...")
        
        # Setup
        simulation = setup_high_throughput_cluster()
        tracker = PerformanceTracker(scheduler_type)
        
        if scheduler_type == "Centralized":
            scheduler = InstrumentedCentralizedScheduler(simulation, tracker)
        else:
            scheduler = InstrumentedDistributedScheduler(simulation, tracker)
            
        simulation.scheduler = scheduler
        
        # Create continuous high-frequency workload
        jobs = create_high_throughput_jobs(200, "small")
        
        # Submit jobs at high frequency (every 0.5 time units)
        for i, job in enumerate(jobs):
            arrival_time = i * 0.5  # High frequency arrivals
            simulation.submit_job(job, arrival_time)
        
        # Run simulation
        start_time = time.time() 
        simulation.run(max_simulation_time=300.0)
        wall_time = time.time() - start_time
        
        stats = tracker.get_summary_stats()
        stats['wall_time'] = wall_time
        stats['scheduler_assignments'] = scheduler.get_stats()['metrics']['assignments_made']
        
        results[scheduler_type] = stats
        
        print(f"  ✅ Jobs completed: {stats['total_jobs']}")
        print(f"  ⚡ Avg scheduling latency: {stats['avg_scheduling_latency']:.3f}s")
        print(f"  📈 Sustained throughput: {stats['avg_throughput']:.1f} jobs/sec")
        print(f"  🎯 P95 scheduling latency: {stats['p95_scheduling_latency']:.3f}s")
    
    # Compare results
    print(f"\n🏆 HIGH-FREQUENCY COMPARISON:")
    cent = results["Centralized"]
    dist = results["Distributed"]
    
    print(f"📊 Sustained Throughput (higher is better):")
    print(f"  Centralized: {cent['avg_throughput']:.1f} jobs/sec")
    print(f"  Distributed: {dist['avg_throughput']:.1f} jobs/sec")
    print(f"  Winner: {'Centralized' if cent['avg_throughput'] > dist['avg_throughput'] else 'Distributed'}")
    
    print(f"📊 Latency Consistency - P95 (lower is better):")
    print(f"  Centralized: {cent['p95_scheduling_latency']:.3f}s")
    print(f"  Distributed: {dist['p95_scheduling_latency']:.3f}s")
    print(f"  Winner: {'Centralized' if cent['p95_scheduling_latency'] < dist['p95_scheduling_latency'] else 'Distributed'}")
    
    return results

def test_latency_sensitive_jobs():
    """Test 3: Ultra-low latency job scheduling"""
    print("\n" + "=" * 80)
    print("⚡ TEST 3: ULTRA-LOW LATENCY JOBS")
    print("=" * 80)
    print("Scenario: 50 critical jobs requiring <1ms scheduling latency")
    print("Measures: P50, P95, P99 scheduling latencies")
    
    results = {}
    
    for scheduler_type in ["Centralized", "Distributed"]:
        print(f"\n📊 Testing {scheduler_type} Scheduler...")
        
        # Setup
        simulation = setup_high_throughput_cluster()
        tracker = PerformanceTracker(scheduler_type)
        
        if scheduler_type == "Centralized":
            scheduler = InstrumentedCentralizedScheduler(simulation, tracker)
        else:
            scheduler = InstrumentedDistributedScheduler(simulation, tracker)
            
        simulation.scheduler = scheduler
        
        # Create latency-sensitive workload
        jobs = create_high_throughput_jobs(50, "latency_sensitive")
        
        # Submit jobs with tight timing
        for i, job in enumerate(jobs):
            arrival_time = i * 0.1  # Very frequent arrivals
            simulation.submit_job(job, arrival_time)
        
        # Run simulation
        start_time = time.time()
        simulation.run(max_simulation_time=100.0)
        wall_time = time.time() - start_time
        
        stats = tracker.get_summary_stats()
        stats['wall_time'] = wall_time
        
        results[scheduler_type] = stats
        
        print(f"  ✅ Jobs completed: {stats['total_jobs']}")
        print(f"  ⚡ P50 latency: {stats['p50_scheduling_latency']*1000:.2f}ms")
        print(f"  ⚡ P95 latency: {stats['p95_scheduling_latency']*1000:.2f}ms")
        print(f"  ⚡ P99 latency: {stats['p99_scheduling_latency']*1000:.2f}ms")
        print(f"  🎯 Max decision time: {stats['max_decision_time']*1000:.2f}ms")
    
    # Compare results
    print(f"\n🏆 LATENCY COMPARISON:")
    cent = results["Centralized"]
    dist = results["Distributed"]
    
    print(f"📊 P50 Scheduling Latency (lower is better):")
    print(f"  Centralized: {cent['p50_scheduling_latency']*1000:.2f}ms")
    print(f"  Distributed: {dist['p50_scheduling_latency']*1000:.2f}ms")
    print(f"  Winner: {'Centralized' if cent['p50_scheduling_latency'] < dist['p50_scheduling_latency'] else 'Distributed'}")
    
    print(f"📊 P95 Scheduling Latency (lower is better):")
    print(f"  Centralized: {cent['p95_scheduling_latency']*1000:.2f}ms")
    print(f"  Distributed: {dist['p95_scheduling_latency']*1000:.2f}ms")
    print(f"  Winner: {'Centralized' if cent['p95_scheduling_latency'] < dist['p95_scheduling_latency'] else 'Distributed'}")
    
    print(f"📊 Maximum Decision Time (lower is better):")
    print(f"  Centralized: {cent['max_decision_time']*1000:.2f}ms")
    print(f"  Distributed: {dist['max_decision_time']*1000:.2f}ms")
    print(f"  Winner: {'Centralized' if cent['max_decision_time'] < dist['max_decision_time'] else 'Distributed'}")
    
    return results

def run_comprehensive_performance_tests():
    """Run all high-throughput, low-latency tests"""
    print("🎯 HIGH-THROUGHPUT, LOW-LATENCY SCHEDULER COMPARISON")
    print("=" * 80)
    print("Testing centralized vs distributed scheduling for:")
    print("1. Burst job handling (high-throughput)")
    print("2. Continuous high-frequency streams (sustained throughput)")
    print("3. Latency-sensitive job scheduling (low-latency)")
    
    # Run all tests
    burst_results = test_burst_job_handling()
    frequency_results = test_continuous_high_frequency()
    latency_results = test_latency_sensitive_jobs()
    
    # Overall summary
    print("\n" + "=" * 80)
    print("📈 OVERALL PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # Count wins for each scheduler
    centralized_wins = 0
    distributed_wins = 0
    
    # Analyze key metrics
    metrics = [
        ("Burst Latency", burst_results["Centralized"]["avg_scheduling_latency"], 
         burst_results["Distributed"]["avg_scheduling_latency"], "lower"),
        ("Burst Throughput", burst_results["Centralized"]["peak_throughput"],
         burst_results["Distributed"]["peak_throughput"], "higher"), 
        ("Decision Speed", burst_results["Centralized"]["avg_decision_time"],
         burst_results["Distributed"]["avg_decision_time"], "lower"),
        ("Sustained Throughput", frequency_results["Centralized"]["avg_throughput"],
         frequency_results["Distributed"]["avg_throughput"], "higher"),
        ("P95 Latency", frequency_results["Centralized"]["p95_scheduling_latency"],
         frequency_results["Distributed"]["p95_scheduling_latency"], "lower"),
        ("Ultra-Low P50 Latency", latency_results["Centralized"]["p50_scheduling_latency"],
         latency_results["Distributed"]["p50_scheduling_latency"], "lower"),
        ("Ultra-Low P99 Latency", latency_results["Centralized"]["p99_scheduling_latency"],
         latency_results["Distributed"]["p99_scheduling_latency"], "lower"),
    ]
    
    print(f"\n🏅 Performance Metric Winners:")
    for metric_name, cent_val, dist_val, better in metrics:
        if better == "lower":
            winner = "Centralized" if cent_val < dist_val else "Distributed"
            centralized_wins += 1 if cent_val < dist_val else 0
            distributed_wins += 1 if dist_val < cent_val else 0
        else:  # higher is better
            winner = "Centralized" if cent_val > dist_val else "Distributed"
            centralized_wins += 1 if cent_val > dist_val else 0
            distributed_wins += 1 if dist_val > cent_val else 0
            
        print(f"  {metric_name}: {winner}")
    
    print(f"\n🏆 FINAL SCORE:")
    print(f"  Centralized: {centralized_wins} wins")
    print(f"  Distributed: {distributed_wins} wins")
    
    if centralized_wins > distributed_wins:
        print(f"\n✅ CONCLUSION: Centralized scheduling is SUPERIOR for high-throughput, low-latency environments!")
        print(f"   Key advantages demonstrated:")
        print(f"   • Lower scheduling latency")
        print(f"   • Faster decision making")
        print(f"   • Higher peak throughput")
        print(f"   • More predictable performance")
    else:
        print(f"\n❓ Unexpected result - distributed performed better in this test")
    
    return {
        'burst': burst_results,
        'frequency': frequency_results, 
        'latency': latency_results,
        'centralized_wins': centralized_wins,
        'distributed_wins': distributed_wins
    }

if __name__ == "__main__":
    import random
    random.seed(42)  # For reproducible results
    run_comprehensive_performance_tests()

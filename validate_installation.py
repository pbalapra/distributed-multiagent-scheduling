#!/usr/bin/env python3
"""
Installation Validation Script for Distributed Multi-Agent Scheduling
===================================================================

This script validates that the package is properly installed and all
components are working correctly before running evaluations.
"""

import sys
import traceback
from pathlib import Path
from typing import List, Tuple, Any

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version meets requirements."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= required_version:
        return True, f"Python {sys.version} (OK)"
    else:
        return False, f"Python {sys.version} (requires >= {required_version[0]}.{required_version[1]})"

def check_imports() -> List[Tuple[str, bool, str]]:
    """Check if all required modules can be imported."""
    modules = [
        # Core dependencies
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib.pyplot'),
        ('pandas', 'pandas'),
        ('seaborn', 'seaborn'),
        
        # Standard library (should always work)
        ('dataclasses', 'dataclasses'),
        ('typing', 'typing'),
        ('datetime', 'datetime'),
        ('threading', 'threading'),
        ('pathlib', 'pathlib'),
        ('json', 'json'),
        ('statistics', 'statistics'),
        ('heapq', 'heapq'),
        ('random', 'random'),
        ('uuid', 'uuid'),
        ('enum', 'enum'),
        ('collections', 'collections'),
        ('time', 'time'),
        ('os', 'os'),
        ('sys', 'sys'),
        
        # Project modules
        ('src.agents.base_agent', 'src.agents.base_agent'),
        ('src.agents.resource_agent', 'src.agents.resource_agent'),
        ('src.communication.protocol', 'src.communication.protocol'),
        ('src.scheduler.discrete_event_scheduler', 'src.scheduler.discrete_event_scheduler'),
        ('src.jobs.job', 'src.jobs.job'),
        ('src.resources.resource', 'src.resources.resource'),
    ]
    
    results = []
    for display_name, import_name in modules:
        try:
            __import__(import_name)
            results.append((display_name, True, "OK"))
        except ImportError as e:
            results.append((display_name, False, str(e)))
        except Exception as e:
            results.append((display_name, False, f"Unexpected error: {e}"))
    
    return results

def check_file_structure() -> List[Tuple[str, bool, str]]:
    """Check if expected files and directories exist."""
    expected_paths = [
        # Core source files
        ('src/__init__.py', 'file'),
        ('src/agents/__init__.py', 'file'),
        ('src/agents/base_agent.py', 'file'),
        ('src/agents/resource_agent.py', 'file'),
        ('src/communication/__init__.py', 'file'),
        ('src/communication/protocol.py', 'file'),
        ('src/scheduler/__init__.py', 'file'),
        ('src/scheduler/discrete_event_scheduler.py', 'file'),
        ('src/jobs/__init__.py', 'file'),
        ('src/jobs/job.py', 'file'),
        ('src/resources/__init__.py', 'file'),
        ('src/resources/resource.py', 'file'),
        
        # Evaluation scripts
        ('evaluation/', 'dir'),
        ('evaluation/systematic_resilience_evaluation.py', 'file'),
        ('evaluation/quick_resilience_test.py', 'file'),
        ('evaluation/ultra_quick_test.py', 'file'),
        ('evaluation/fault_tolerant_test.py', 'file'),
        ('evaluation/high_throughput_test.py', 'file'),
        
        # Configuration files
        ('requirements.txt', 'file'),
        ('setup.py', 'file'),
        ('README.md', 'file'),
        ('LICENSE', 'file'),
        ('CONTRIBUTING.md', 'file'),
        
        # Figure generation scripts
        ('create_bw_publication_figures.py', 'file'),
        ('create_publication_figures.py', 'file'),
        ('resilience_evaluation_results.tex', 'file'),
        
        # Documentation
        ('docs/', 'dir'),
        ('bw_figure_descriptions.md', 'file'),
        ('figure_descriptions.md', 'file'),
    ]
    
    results = []
    for path_str, path_type in expected_paths:
        path = Path(path_str)
        
        if path_type == 'file':
            if path.is_file():
                size = path.stat().st_size
                results.append((path_str, True, f"File exists ({size:,} bytes)"))
            else:
                results.append((path_str, False, "File not found"))
        
        elif path_type == 'dir':
            if path.is_dir():
                file_count = len(list(path.iterdir()))
                results.append((path_str, True, f"Directory exists ({file_count} items)"))
            else:
                results.append((path_str, False, "Directory not found"))
    
    return results

def test_basic_functionality() -> List[Tuple[str, bool, str]]:
    """Test basic functionality of core components."""
    results = []
    
    # Test 1: Import and create basic objects
    try:
        from src.resources.resource import Resource, ResourceCapacity, ResourceType
        
        capacity = ResourceCapacity(total_cpu_cores=8, total_memory_gb=16)
        resource = Resource("test-resource", "Test Resource", ResourceType.CPU_CLUSTER,
                          capacity, "test-location", 10.0)
        
        results.append(("Resource Creation", True, "Successfully created resource objects"))
    except Exception as e:
        results.append(("Resource Creation", False, f"Failed: {e}"))
    
    # Test 2: Create job
    try:
        from src.jobs.job import Job, JobPriority, ResourceRequirement
        
        requirements = ResourceRequirement(cpu_cores=2, memory_gb=4)
        job = Job("test-job", "Test Job", JobPriority.MEDIUM, requirements, 10.0, 1.0)
        
        results.append(("Job Creation", True, "Successfully created job objects"))
    except Exception as e:
        results.append(("Job Creation", False, f"Failed: {e}"))
    
    # Test 3: Create message bus
    try:
        from src.communication.protocol import MessageBus
        
        message_bus = MessageBus()
        results.append(("Message Bus", True, "Successfully created message bus"))
    except Exception as e:
        results.append(("Message Bus", False, f"Failed: {e}"))
    
    # Test 4: Create scheduler
    try:
        from src.scheduler.discrete_event_scheduler import DiscreteEventScheduler
        
        scheduler = DiscreteEventScheduler()
        results.append(("Scheduler Creation", True, "Successfully created scheduler"))
    except Exception as e:
        results.append(("Scheduler Creation", False, f"Failed: {e}"))
    
    # Test 5: Test figure generation imports
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        # Create simple test plot
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar([1, 2, 3], [10, 20, 15])
        ax.set_title("Test Plot")
        plt.close(fig)
        
        results.append(("Figure Generation", True, "Successfully created test plot"))
    except Exception as e:
        results.append(("Figure Generation", False, f"Failed: {e}"))
    
    return results

def run_mini_evaluation() -> Tuple[bool, str]:
    """Run a minimal evaluation to test the framework."""
    try:
        from evaluation.systematic_resilience_evaluation import (
            ExperimentConfig, ScalableResilienceTracker, 
            create_scalable_cluster, create_scalable_jobs
        )
        
        # Create minimal test configuration
        config = ExperimentConfig(
            name="ValidationTest",
            num_jobs=5,
            num_agents=2,
            agent_failure_rate=0.1,
            scheduler_failure_rate=0.05,
            job_arrival_pattern='constant',
            failure_pattern='random',
            simulation_time=10.0,
            repetitions=1
        )
        
        # Create tracker and simulation
        tracker = ScalableResilienceTracker("validation-test")
        simulation = create_scalable_cluster(config.num_agents, config.agent_failure_rate, tracker)
        
        # Create and submit jobs
        jobs_with_times = create_scalable_jobs(
            config.num_jobs, config.job_arrival_pattern, config.simulation_time)
        
        if len(jobs_with_times) == config.num_jobs:
            return True, f"Successfully created {len(jobs_with_times)} jobs and simulation"
        else:
            return False, f"Expected {config.num_jobs} jobs, got {len(jobs_with_times)}"
            
    except Exception as e:
        return False, f"Mini evaluation failed: {e}"

def print_results(title: str, results: List[Tuple[str, bool, str]]):
    """Print formatted results."""
    print(f"\n{title}")
    print("-" * len(title))
    
    passed = 0
    total = len(results)
    
    for name, success, message in results:
        status = "✅" if success else "❌"
        print(f"{status} {name:30} | {message}")
        if success:
            passed += 1
    
    print(f"\nResult: {passed}/{total} passed ({passed/total:.1%})")
    return passed == total

def main():
    """Run complete installation validation."""
    print("🔍 INSTALLATION VALIDATION")
    print("=" * 60)
    print("Validating distributed multi-agent scheduling installation...")
    
    all_good = True
    
    # Check Python version
    print("\n🐍 PYTHON VERSION")
    print("-" * 15)
    python_ok, python_msg = check_python_version()
    status = "✅" if python_ok else "❌"
    print(f"{status} {python_msg}")
    all_good &= python_ok
    
    # Check imports
    import_results = check_imports()
    import_ok = print_results("📦 MODULE IMPORTS", import_results)
    all_good &= import_ok
    
    # Check file structure
    file_results = check_file_structure()
    file_ok = print_results("📂 FILE STRUCTURE", file_results)
    all_good &= file_ok
    
    # Test basic functionality
    func_results = test_basic_functionality()
    func_ok = print_results("⚙️  BASIC FUNCTIONALITY", func_results)
    all_good &= func_ok
    
    # Run mini evaluation
    print("\n🧪 MINI EVALUATION")
    print("-" * 15)
    eval_ok, eval_msg = run_mini_evaluation()
    status = "✅" if eval_ok else "❌"
    print(f"{status} {eval_msg}")
    all_good &= eval_ok
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    if all_good:
        print("🎉 ALL CHECKS PASSED!")
        print("\n✅ Installation is valid and ready for evaluation")
        print("💡 Next steps:")
        print("   - Run quick test: python run_evaluation.py --quick")
        print("   - Run full evaluation: python run_evaluation.py --full")
        print("   - See README.md for detailed usage instructions")
        
    else:
        print("❌ VALIDATION FAILED!")
        print("\n🔧 Installation issues detected:")
        
        if not python_ok:
            print("   - Upgrade Python to version 3.8 or higher")
        
        if not import_ok:
            print("   - Install missing dependencies: pip install -r requirements.txt")
        
        if not file_ok:
            print("   - Ensure all project files are present")
            print("   - Re-clone repository if files are missing")
        
        if not func_ok:
            print("   - Check for import/dependency conflicts")
        
        if not eval_ok:
            print("   - Verify evaluation framework components")
        
        print("\n📞 Need help?")
        print("   - Check README.md for installation instructions")
        print("   - Report issues: https://github.com/username/distributed-multiagent-scheduling/issues")
    
    # Exit with appropriate code
    sys.exit(0 if all_good else 1)

if __name__ == "__main__":
    main()
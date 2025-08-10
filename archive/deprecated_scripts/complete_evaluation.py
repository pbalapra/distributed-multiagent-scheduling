#!/usr/bin/env python3
"""
Complete Evaluation of LLM-Enhanced Multiagent Scheduling System
===============================================================

This script runs a comprehensive evaluation of all system components,
including LLM integration, fault tolerance, scalability, and performance.
"""

import sys
import os
import json
import time
import subprocess
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_command_safe(cmd: str) -> Dict[str, Any]:
    """Run a command safely and return results"""
    try:
        start_time = time.time()
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=300)
        duration = time.time() - start_time
        
        return {
            "command": cmd,
            "success": result.returncode == 0,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "command": cmd,
            "success": False,
            "duration": 300,
            "stdout": "",
            "stderr": "Command timed out after 300 seconds",
            "exit_code": -1
        }
    except Exception as e:
        return {
            "command": cmd,
            "success": False,
            "duration": 0,
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1
        }

def evaluate_llm_integration():
    """Test LLM integration capabilities"""
    print("🧠 Testing LLM Integration...")
    print("-" * 50)
    
    # Test Ollama integration
    ollama_result = run_command_safe("python test_ollama_integration.py")
    
    # Test LLM agent demo
    demo_result = run_command_safe("python demos/llm_agent_demo.py")
    
    results = {
        "ollama_integration": {
            "success": ollama_result["success"],
            "duration": ollama_result["duration"]
        },
        "llm_demo": {
            "success": demo_result["success"], 
            "duration": demo_result["duration"]
        }
    }
    
    # Parse demo output for specific metrics
    if demo_result["success"]:
        output = demo_result["stdout"]
        # Extract key performance indicators
        if "✅ Using Ollama Provider" in output:
            results["ollama_provider_available"] = True
        if "LLM Score:" in output:
            results["job_scoring_working"] = True
        if "fault recovery decision:" in output:
            results["fault_recovery_working"] = True
        if "negotiation decision:" in output:
            results["negotiation_working"] = True
    
    success_rate = sum(1 for r in results.values() if isinstance(r, dict) and r.get("success", False)) / 2
    
    print(f"🧠 LLM Integration Results:")
    print(f"   Ollama Integration: {'✅' if results['ollama_integration']['success'] else '❌'}")
    print(f"   Agent Demo: {'✅' if results['llm_demo']['success'] else '❌'}")
    print(f"   Success Rate: {success_rate:.1%}")
    
    return results

def evaluate_discrete_event_simulation():
    """Test discrete event simulation capabilities"""
    print("\n⚡ Testing Discrete Event Simulation...")
    print("-" * 50)
    
    # Test main discrete event demo
    sim_result = run_command_safe("python true_discrete_event_demo.py")
    
    results = {
        "simulation": {
            "success": sim_result["success"],
            "duration": sim_result["duration"]
        }
    }
    
    if sim_result["success"]:
        output = sim_result["stdout"]
        
        # Extract performance metrics
        if "Speed ratio:" in output:
            import re
            speed_match = re.search(r"Speed ratio: ([\d,]+\.?\d*)x", output)
            if speed_match:
                speed_ratio = float(speed_match.group(1).replace(",", ""))
                results["speed_ratio"] = speed_ratio
        
        if "Jobs completed:" in output:
            completed_match = re.search(r"Jobs completed: (\d+)", output)
            if completed_match:
                results["jobs_completed"] = int(completed_match.group(1))
        
        if "Success rate:" in output:
            success_match = re.search(r"Success rate: ([\d.]+)%", output)
            if success_match:
                results["job_success_rate"] = float(success_match.group(1))
    
    print(f"⚡ Discrete Event Results:")
    print(f"   Simulation: {'✅' if results['simulation']['success'] else '❌'}")
    if "speed_ratio" in results:
        print(f"   Speed Ratio: {results['speed_ratio']:,.0f}x faster than real-time")
    if "jobs_completed" in results:
        print(f"   Jobs Completed: {results['jobs_completed']}")
    if "job_success_rate" in results:
        print(f"   Success Rate: {results['job_success_rate']:.1f}%")
    
    return results

def evaluate_scheduling_comparison():
    """Test centralized vs distributed scheduling"""
    print("\n🔄 Testing Scheduling Algorithms...")
    print("-" * 50)
    
    # Test combined scheduling demo
    sched_result = run_command_safe("python demos/combined_scheduling_demo.py")
    
    results = {
        "scheduling_comparison": {
            "success": sched_result["success"],
            "duration": sched_result["duration"]
        }
    }
    
    if sched_result["success"]:
        output = sched_result["stdout"]
        
        # Extract scheduling metrics
        centralized_jobs = distributed_jobs = 0
        centralized_time = distributed_time = 0
        
        import re
        
        # Centralized results
        cent_jobs = re.search(r"Centralized:\s+(\d+) jobs assigned", output)
        if cent_jobs:
            centralized_jobs = int(cent_jobs.group(1))
            
        cent_time = re.search(r"Centralized:\s+.*?(\d+\.?\d*)ms average", output)
        if cent_time:
            centralized_time = float(cent_time.group(1))
        
        # Distributed results  
        dist_jobs = re.search(r"Distributed:\s+(\d+) jobs assigned", output)
        if dist_jobs:
            distributed_jobs = int(dist_jobs.group(1))
            
        dist_time = re.search(r"Distributed:\s+.*?(\d+\.?\d*)ms average", output)
        if dist_time:
            distributed_time = float(dist_time.group(1))
        
        # Messages sent
        messages = re.search(r"Messages sent: (\d+)", output)
        if messages:
            results["distributed_messages"] = int(messages.group(1))
        
        results.update({
            "centralized_jobs": centralized_jobs,
            "distributed_jobs": distributed_jobs,
            "centralized_decision_time": centralized_time,
            "distributed_decision_time": distributed_time
        })
    
    print(f"🔄 Scheduling Results:")
    print(f"   Comparison: {'✅' if results['scheduling_comparison']['success'] else '❌'}")
    if "centralized_jobs" in results:
        print(f"   Centralized Jobs: {results['centralized_jobs']}")
        print(f"   Distributed Jobs: {results['distributed_jobs']}")
    if "distributed_messages" in results:
        print(f"   Distributed Messages: {results['distributed_messages']}")
    
    return results

def evaluate_system_resilience():
    """Test system resilience and fault tolerance"""
    print("\n🛡️ Testing System Resilience...")
    print("-" * 50)
    
    # Test ultra-quick evaluation
    resilience_result = run_command_safe("python evaluation/ultra_quick_test.py")
    
    # Test evaluation framework
    framework_result = run_command_safe("python evaluation/evaluation_framework.py")
    
    results = {
        "resilience_test": {
            "success": resilience_result["success"],
            "duration": resilience_result["duration"]
        },
        "framework_test": {
            "success": framework_result["success"],
            "duration": framework_result["duration"]
        }
    }
    
    if resilience_result["success"]:
        output = resilience_result["stdout"]
        
        # Extract resilience metrics
        if "Recovery time" in output:
            import re
            recovery_matches = re.findall(r"Recovery time ([\d.]+)s", output)
            if recovery_matches:
                results["avg_recovery_time"] = statistics.mean(map(float, recovery_matches))
        
        if "Success rate:" in output:
            success_matches = re.findall(r"Success rate: ([\d.]+)%", output)
            if success_matches:
                results["resilience_success_rate"] = statistics.mean(map(float, success_matches))
        
        if "Throughput" in output:
            throughput_matches = re.findall(r"Throughput ([\d.]+) jobs/s", output)
            if throughput_matches:
                results["avg_throughput"] = statistics.mean(map(float, throughput_matches))
    
    success_rate = sum(1 for r in [results["resilience_test"], results["framework_test"]] if r["success"]) / 2
    
    print(f"🛡️ Resilience Results:")
    print(f"   Resilience Test: {'✅' if results['resilience_test']['success'] else '❌'}")
    print(f"   Framework Test: {'✅' if results['framework_test']['success'] else '❌'}")
    print(f"   Overall Success: {success_rate:.1%}")
    if "avg_recovery_time" in results:
        print(f"   Avg Recovery Time: {results['avg_recovery_time']:.1f}s")
    if "resilience_success_rate" in results:
        print(f"   Resilience Success Rate: {results['resilience_success_rate']:.1f}%")
    
    return results

def generate_comprehensive_report(results: Dict[str, Any]):
    """Generate comprehensive evaluation report"""
    print("\n📊 COMPREHENSIVE EVALUATION REPORT")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Generated: {timestamp}")
    
    # Calculate overall metrics
    total_tests = 0
    successful_tests = 0
    total_duration = 0
    
    for category, data in results.items():
        print(f"\n🔍 {category.upper().replace('_', ' ')}:")
        for test_name, test_data in data.items():
            if isinstance(test_data, dict) and "success" in test_data:
                total_tests += 1
                if test_data["success"]:
                    successful_tests += 1
                total_duration += test_data.get("duration", 0)
                
                status = "✅ PASS" if test_data["success"] else "❌ FAIL"
                duration = test_data.get("duration", 0)
                print(f"   {status} {test_name}: {duration:.2f}s")
            elif isinstance(test_data, (int, float, bool)):
                print(f"   📈 {test_name}: {test_data}")
    
    # Overall summary
    overall_success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n🎯 OVERALL SUMMARY:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Successful: {successful_tests}")
    print(f"   Failed: {total_tests - successful_tests}")
    print(f"   Success Rate: {overall_success_rate:.1f}%")
    print(f"   Total Duration: {total_duration:.2f}s")
    
    # Key achievements
    print(f"\n🏆 KEY ACHIEVEMENTS:")
    
    if results.get("llm_results", {}).get("ollama_integration", {}).get("success"):
        print("   ✅ LLM Integration with Ollama working")
    
    if "speed_ratio" in results.get("simulation_results", {}):
        speed = results["simulation_results"]["speed_ratio"]
        print(f"   ⚡ Discrete event simulation: {speed:,.0f}x faster than real-time")
    
    if results.get("scheduling_results", {}).get("distributed_messages", 0) > 0:
        msgs = results["scheduling_results"]["distributed_messages"]
        print(f"   🌐 Distributed coordination: {msgs} messages processed")
    
    if "resilience_success_rate" in results.get("resilience_results", {}):
        rate = results["resilience_results"]["resilience_success_rate"]
        print(f"   🛡️ System resilience: {rate:.1f}% success under failures")
    
    # Performance benchmarks
    print(f"\n📈 PERFORMANCE BENCHMARKS:")
    
    if "avg_throughput" in results.get("resilience_results", {}):
        throughput = results["resilience_results"]["avg_throughput"]
        print(f"   📊 Average Throughput: {throughput:.2f} jobs/sec")
    
    if "centralized_jobs" in results.get("scheduling_results", {}):
        cent = results["scheduling_results"]["centralized_jobs"]
        dist = results["scheduling_results"]["distributed_jobs"]
        print(f"   📋 Job Assignment: Centralized={cent}, Distributed={dist}")
    
    if "avg_recovery_time" in results.get("resilience_results", {}):
        recovery = results["resilience_results"]["avg_recovery_time"]
        print(f"   🔧 Fault Recovery: {recovery:.1f}s average")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if overall_success_rate >= 80:
        print("   🎉 System is production-ready with excellent performance")
        print("   🚀 Consider deployment in high-throughput environments")
        print("   🔄 Monitor LLM integration performance in production")
    elif overall_success_rate >= 60:
        print("   ⚠️ System shows good potential but needs optimization")
        print("   🔧 Focus on fixing failing test components")
        print("   📊 Improve monitoring and error handling")
    else:
        print("   ❌ System needs significant improvements before deployment")
        print("   🛠️ Address critical failures before production use")
        print("   📋 Review architecture and implementation")
    
    print(f"\n📁 Detailed logs and results available in evaluation output")
    
    return {
        "timestamp": timestamp,
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": overall_success_rate,
        "total_duration": total_duration,
        "results": results
    }

def save_results(report_data: Dict[str, Any]):
    """Save evaluation results to files"""
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save JSON report
    json_file = results_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {json_file}")
    
    return json_file

def main():
    """Run complete evaluation"""
    print("🚀 COMPLETE LLM-ENHANCED MULTIAGENT SYSTEM EVALUATION")
    print("=" * 80)
    print("This comprehensive evaluation tests all system components")
    print("including LLM integration, scheduling algorithms, fault tolerance,")
    print("and performance characteristics.\n")
    
    start_time = time.time()
    
    # Run all evaluation components
    results = {}
    
    try:
        results["llm_results"] = evaluate_llm_integration()
        results["simulation_results"] = evaluate_discrete_event_simulation()
        results["scheduling_results"] = evaluate_scheduling_comparison()
        results["resilience_results"] = evaluate_system_resilience()
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        return 1
    
    # Generate comprehensive report
    report_data = generate_comprehensive_report(results)
    
    # Save results
    save_results(report_data)
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Complete evaluation finished in {total_time:.1f}s")
    
    # Return appropriate exit code
    return 0 if report_data["success_rate"] >= 70 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

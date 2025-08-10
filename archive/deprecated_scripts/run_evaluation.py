#!/usr/bin/env python3
"""
Automated Evaluation Runner for Distributed Multi-Agent Scheduling
================================================================

This script provides a unified interface for running all evaluation scenarios
and generating publication-ready results with proper reproducibility controls.
"""

import sys
import time
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional

def run_command(cmd: List[str], description: str, timeout: Optional[int] = None) -> bool:
    """
    Run a command with proper error handling and output capture.
    
    Args:
        cmd: Command and arguments to run
        description: Human-readable description of the command
        timeout: Maximum execution time in seconds (None for no timeout)
        
    Returns:
        bool: True if command succeeded, False otherwise
    """
    print(f"\n🔄 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"   ✅ Completed in {elapsed:.1f}s")
        
        # Show last few lines of output if available
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 3:
                print(f"   📄 Output (last 3 lines):")
                for line in lines[-3:]:
                    print(f"      {line}")
            else:
                print(f"   📄 Output: {result.stdout.strip()}")
        
        return True
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"   ⏰ Timeout after {elapsed:.1f}s (limit: {timeout}s)")
        return False
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"   ❌ Failed after {elapsed:.1f}s (exit code: {e.returncode})")
        if e.stderr:
            print(f"   📄 Error: {e.stderr.strip()}")
        return False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   💥 Exception after {elapsed:.1f}s: {e}")
        return False

def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_modules = [
        'numpy', 'matplotlib', 'pandas', 'seaborn'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module} - OK")
        except ImportError:
            print(f"   ❌ {module} - MISSING")
            return False
    
    return True

def run_quick_evaluation() -> bool:
    """Run quick evaluation for immediate validation."""
    return run_command(
        ['python', 'evaluation/ultra_quick_test.py'],
        "Running ultra-quick evaluation (30 seconds)",
        timeout=120
    )

def run_systematic_evaluation() -> bool:
    """Run complete systematic resilience evaluation."""
    return run_command(
        ['python', 'evaluation/systematic_resilience_evaluation.py'],
        "Running systematic resilience evaluation (30-45 minutes)",
        timeout=3600  # 1 hour timeout
    )

def run_fault_tolerance_tests() -> bool:
    """Run fault tolerance specific tests."""
    return run_command(
        ['python', 'evaluation/fault_tolerant_test.py'],
        "Running fault tolerance tests (10-15 minutes)",
        timeout=1800  # 30 minutes timeout
    )

def run_high_throughput_tests() -> bool:
    """Run high throughput performance tests."""
    return run_command(
        ['python', 'evaluation/high_throughput_test.py'],
        "Running high throughput tests (5-10 minutes)",
        timeout=900  # 15 minutes timeout
    )

def generate_figures() -> bool:
    """Generate publication-ready figures."""
    success = True
    
    # Generate black & white figures
    if not run_command(
        ['python', 'create_bw_publication_figures.py'],
        "Generating black & white publication figures",
        timeout=300
    ):
        success = False
    
    # Generate color figures
    if not run_command(
        ['python', 'create_publication_figures.py'],
        "Generating color publication figures",
        timeout=300
    ):
        success = False
    
    return success

def compile_latex_report() -> bool:
    """Compile LaTeX results document."""
    latex_file = "resilience_evaluation_results.tex"
    
    if not Path(latex_file).exists():
        print(f"   ⚠️  LaTeX file {latex_file} not found - skipping compilation")
        return True
    
    # Check if pdflatex is available
    try:
        subprocess.run(['pdflatex', '--version'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ⚠️  pdflatex not found - skipping LaTeX compilation")
        print("   💡 Install LaTeX distribution to generate PDF report")
        return True
    
    # Compile LaTeX document
    return run_command(
        ['pdflatex', latex_file],
        "Compiling LaTeX results document",
        timeout=120
    )

def validate_results() -> bool:
    """Validate that expected output files were generated."""
    print("\n🔍 Validating generated results...")
    
    expected_files = [
        'bw_figures/figure1_scale_job_count.png',
        'bw_figures/figure11_win_rates.png',
        'figures/figure1_scale_testing.png',
        'figures/table1_statistical_summary.png',
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            file_size = Path(file_path).stat().st_size
            print(f"   ✅ {file_path} ({file_size:,} bytes)")
    
    if missing_files:
        print(f"   ❌ Missing files:")
        for file_path in missing_files:
            print(f"      - {file_path}")
        return False
    
    # Check for results JSON files
    json_files = list(Path('.').glob('resilience_study_results_*.json'))
    if json_files:
        latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
        file_size = latest_json.stat().st_size
        print(f"   ✅ {latest_json} ({file_size:,} bytes)")
    else:
        print(f"   ⚠️  No resilience study JSON results found")
    
    return True

def print_summary(results: Dict[str, bool], total_time: float):
    """Print evaluation summary."""
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    
    total_steps = len(results)
    successful_steps = sum(results.values())
    
    print(f"Total Steps: {total_steps}")
    print(f"Successful: {successful_steps}")
    print(f"Failed: {total_steps - successful_steps}")
    print(f"Success Rate: {successful_steps/total_steps:.1%}")
    print(f"Total Time: {total_time/60:.1f} minutes")
    
    print(f"\n📋 Step Results:")
    for step, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {step}")
    
    if all(results.values()):
        print(f"\n🎉 ALL EVALUATIONS COMPLETED SUCCESSFULLY!")
        print(f"   📊 Results available in bw_figures/ and figures/")
        print(f"   📄 Raw data in resilience_study_results_*.json")
        print(f"   📋 See README.md for result interpretation")
    else:
        print(f"\n⚠️  Some evaluations failed - check output above for details")
        print(f"   💡 You can re-run individual components using specific scripts")

def main():
    """Main evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Automated evaluation runner for distributed multi-agent scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py --quick          # Quick validation (2-3 minutes)
  python run_evaluation.py --systematic     # Systematic evaluation (45 minutes)  
  python run_evaluation.py --full           # Complete evaluation (60-90 minutes)
  python run_evaluation.py --figures-only   # Generate figures only
        """
    )
    
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick evaluation only (2-3 minutes)'
    )
    parser.add_argument(
        '--systematic', action='store_true', 
        help='Run systematic evaluation only (30-45 minutes)'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Run complete evaluation suite (60-90 minutes)'
    )
    parser.add_argument(
        '--figures-only', action='store_true',
        help='Generate publication figures only'
    )
    parser.add_argument(
        '--skip-latex', action='store_true',
        help='Skip LaTeX compilation step'
    )
    parser.add_argument(
        '--no-validate', action='store_true',
        help='Skip result validation'
    )
    
    args = parser.parse_args()
    
    # Default to full evaluation if no specific mode selected
    if not any([args.quick, args.systematic, args.full, args.figures_only]):
        args.full = True
    
    print("🚀 DISTRIBUTED MULTI-AGENT SCHEDULING EVALUATION")
    print("="*80)
    
    start_time = time.time()
    results = {}
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed!")
        print("   💡 Run: pip install -r requirements.txt")
        sys.exit(1)
    
    results['Dependencies'] = True
    
    # Run evaluations based on selected mode
    if args.figures_only:
        print("\n📊 FIGURES-ONLY MODE")
        results['Generate Figures'] = generate_figures()
        
    elif args.quick:
        print("\n⚡ QUICK EVALUATION MODE (2-3 minutes)")
        results['Quick Evaluation'] = run_quick_evaluation()
        results['Generate Figures'] = generate_figures()
        
    elif args.systematic:
        print("\n🧪 SYSTEMATIC EVALUATION MODE (30-45 minutes)")
        results['Systematic Evaluation'] = run_systematic_evaluation()
        results['Generate Figures'] = generate_figures()
        
    else:  # Full evaluation
        print("\n🎯 FULL EVALUATION MODE (60-90 minutes)")
        results['Quick Evaluation'] = run_quick_evaluation()
        results['Systematic Evaluation'] = run_systematic_evaluation()
        results['Fault Tolerance Tests'] = run_fault_tolerance_tests()
        results['High Throughput Tests'] = run_high_throughput_tests()
        results['Generate Figures'] = generate_figures()
    
    # Optional steps
    if not args.skip_latex:
        results['Compile LaTeX'] = compile_latex_report()
    
    if not args.no_validate:
        results['Validate Results'] = validate_results()
    
    # Print summary
    total_time = time.time() - start_time
    print_summary(results, total_time)
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
# Multi-Job Scheduling Visualization Guide

This guide explains how to visualize and analyze the results from your multi-agent job scheduling system.

## 🎨 Available Visualization Tools

### 1. **Comprehensive Visualizer** (`visualize_results.py`)
Advanced visualization with real-time capabilities and detailed analysis.

### 2. **Simple Plot Generator** (`plot_results.py`) 
Quick and easy plotting for basic analysis.

### 3. **Enhanced Demo with Visualization** (`multi_job_demo_with_viz.py`)
Live demo with integrated real-time visualization.

---

## 🚀 Quick Start

### Generate Basic Plots from Demo Data
```bash
# Simple plots using demo data
python3 plot_results.py --demo

# Show plots on screen
python3 plot_results.py --demo --show

# Comprehensive analysis with demo data  
python3 visualize_results.py --demo
```

### Plot Custom Data
```bash
# Format: completed_jobs,retries for each agent
python3 plot_results.py --custom 12,2 9,3 14,1 13,1 15,1 --show
```

---

## 📊 Generated Visualizations

### Agent Performance Analysis
- **Jobs Completed by Agent**: Bar chart showing productivity
- **Retries by Agent**: Failure/retry analysis
- **Efficiency Comparison**: Success rates and reliability
- **Resource Utilization**: CPU, memory, GPU usage over time

### System Timeline
- **Job States Over Time**: Pending, running, completed jobs
- **System Load**: Running jobs and capacity utilization
- **Cumulative Retries**: Failure patterns over time

### Resource Analysis
- **Utilization Heatmap**: Resource usage by agent and type
- **Cost Efficiency**: Cost per job analysis
- **Load Balancing**: Job distribution across agents

### Summary Reports
- **Performance Statistics**: Completion rates, efficiency metrics
- **Agent Comparison**: Best and worst performing agents
- **System Health**: Overall success rates and reliability

---

## 🛠️ Usage Examples

### 1. Basic Demo Visualization
```bash
# Generate simple plots from previous demo run
python3 plot_results.py --demo --show
```
**Output**: 
- `simple_plots/agent_comparison.png` - Agent performance comparison
- `simple_plots/system_timeline.png` - Timeline analysis  
- `simple_plots/resource_usage.png` - Resource utilization

### 2. Comprehensive Analysis
```bash
# Full analysis with advanced plots
python3 visualize_results.py --demo
```
**Output**:
- `plots/agent_performance.png` - Detailed agent metrics
- `plots/system_timeline.png` - System performance over time
- `plots/job_distribution.png` - Job distribution and success rates
- `plots/resource_heatmap.png` - Resource utilization heatmap
- `plots/analysis_report.txt` - Detailed text report

### 3. Custom Data Analysis
```bash
# Plot your own results (format: completed,retries)
python3 plot_results.py --custom 15,1 12,2 18,0 10,3 8,4
```

### 4. Live Demo with Visualization
```bash
# Run enhanced demo with real-time plots
python3 multi_job_demo_with_viz.py
```

---

## 📈 Plot Types Explained

### 1. **Agent Comparison Plot**
- **Purpose**: Compare agent performance side-by-side
- **Metrics**: Jobs completed, retries, efficiency percentages
- **Use**: Identify best/worst performing agents

### 2. **System Timeline**
- **Purpose**: Show system behavior over time
- **Metrics**: Pending, running, completed jobs by timestamp
- **Use**: Analyze load patterns and completion rates

### 3. **Resource Utilization Heatmap**
- **Purpose**: Visualize resource usage across agents
- **Metrics**: CPU, Memory, GPU, Network utilization %
- **Use**: Identify resource bottlenecks

### 4. **Job Distribution Pie Charts**
- **Purpose**: Show how jobs are distributed
- **Metrics**: Jobs by priority, jobs by agent
- **Use**: Analyze workload balance

### 5. **Cost Analysis**
- **Purpose**: Economic efficiency analysis
- **Metrics**: Cost per job, total costs by agent
- **Use**: Optimize resource allocation costs

---

## 🔧 Customization Options

### Command Line Arguments

#### `visualize_results.py`
```bash
--demo              # Use simulated demo data
--output-dir DIR    # Custom output directory (default: plots)  
--no-save          # Don't save plots to files
```

#### `plot_results.py`
```bash
--demo             # Generate from demo data
--custom DATA      # Custom data points (completed,retries format)
--show             # Display plots on screen
--no-save         # Don't save plots to files
```

### Integration with Your Code

```python
from visualize_results import JobSchedulingVisualizer

# Create visualizer
visualizer = JobSchedulingVisualizer(save_plots=True, output_dir="my_plots")

# Update agent stats
visualizer.update_agent_stats("agent-1", completed=15, failed=1, retries=2)

# Add timeline events
visualizer.add_timeline_point(datetime.now(), "agent-1", "complete", "job-123")

# Generate all plots
plots = visualizer.create_all_plots()
```

---

## 📁 Output Directory Structure

```
project/
├── plots/                          # Comprehensive visualizations
│   ├── agent_performance.png       # Detailed agent analysis
│   ├── system_timeline.png         # System metrics over time
│   ├── job_distribution.png        # Job and priority analysis
│   ├── resource_heatmap.png         # Resource utilization
│   └── analysis_report.txt         # Summary report
├── simple_plots/                   # Quick visualizations
│   ├── agent_comparison.png        # Basic agent comparison
│   ├── system_timeline.png         # Simple timeline
│   └── resource_usage.png          # Resource usage
└── custom_plots/                   # Custom data plots
    ├── agent_comparison.png
    └── resource_usage.png
```

---

## 🎯 Best Practices

### 1. **Regular Monitoring**
- Generate plots after each demo run
- Compare performance across different configurations
- Track improvements over time

### 2. **Performance Analysis**
- Look for agents with high retry rates (potential issues)
- Identify underutilized resources
- Monitor cost efficiency

### 3. **System Optimization**
- Use resource heatmaps to balance loads
- Analyze timeline plots for bottlenecks
- Compare agent efficiency for tuning

### 4. **Troubleshooting**
- High retry rates → Check agent reliability
- Uneven job distribution → Review scheduling algorithm
- Low resource utilization → Adjust job requirements

---

## 🐛 Troubleshooting

### Common Issues

**"Visualization not available" Error**
```bash
# Install required packages
pip3 install matplotlib seaborn pandas numpy
```

**Empty or Missing Plots**
- Check that demo ran successfully
- Verify data files exist
- Ensure output directory has write permissions

**Custom Data Format Error**
```bash
# Correct format: completed_jobs,retries
python3 plot_results.py --custom 12,2 9,3 14,1
```

### Dependencies
```bash
# Required packages
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.20.0
```

---

## 🚀 Advanced Features

### Real-time Visualization
- Live updating plots during demo execution
- Real-time system metrics tracking
- Dynamic resource utilization monitoring

### Custom Metrics
- Add your own performance indicators
- Create domain-specific visualizations
- Export data for external analysis

### Batch Analysis
- Process multiple demo runs
- Compare different configurations
- Generate performance trends

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure input data format is correct
4. Review the command line arguments

---

**Happy Visualizing! 🎨📊**

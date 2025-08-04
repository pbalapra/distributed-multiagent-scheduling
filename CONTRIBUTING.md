# Contributing to Distributed Multi-Agent Scheduling

We welcome contributions from the research and open-source community! This document provides guidelines for contributing to the project.

## 🤝 Ways to Contribute

### Research Contributions
- **New evaluation scenarios** and failure patterns
- **Performance optimizations** for large-scale simulations
- **Additional scheduling algorithms** for comparison studies
- **Novel metrics** for resilience and performance assessment

### Technical Contributions
- **Bug fixes** and code improvements
- **Documentation** enhancements and tutorials
- **Visualization tools** and interactive dashboards
- **Testing framework** improvements

### Community Contributions
- **Issue reporting** and feature requests
- **Code reviews** and feedback
- **Educational content** and use case examples
- **Integration guides** for other HPC systems

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/your-username/distributed-multiagent-scheduling.git
cd distributed-multiagent-scheduling
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies**
```bash
pip install -r requirements-dev.txt
pip install -e .
```

4. **Install pre-commit hooks**
```bash
pre-commit install
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/evaluation/
```

## 📝 Contribution Guidelines

### Code Style

We follow Python best practices and use automated tools:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black src/ evaluation/ tests/

# Sort imports
isort src/ evaluation/ tests/

# Lint code
flake8 src/ evaluation/ tests/

# Type check
mypy src/
```

### Git Workflow

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
- Write clear, documented code
- Add tests for new functionality
- Update documentation as needed

3. **Test your changes**
```bash
python -m pytest
```

4. **Commit with clear messages**
```bash
git add .
git commit -m "Add: Brief description of your changes"
```

5. **Push and create pull request**
```bash
git push origin feature/your-feature-name
```

### Commit Message Convention

Use clear, descriptive commit messages:

- **Add**: New features or functionality
- **Fix**: Bug fixes
- **Update**: Improvements to existing features
- **Refactor**: Code restructuring without functional changes
- **Docs**: Documentation updates
- **Test**: Test additions or improvements

Examples:
```
Add: Competitive bidding algorithm with priority weighting
Fix: Race condition in agent heartbeat monitoring
Update: Improve fault tolerance score calculation
Docs: Add installation guide for Windows users
```

## 🧪 Testing Guidelines

### Writing Tests

- **Unit tests** for individual functions and classes
- **Integration tests** for component interactions
- **Evaluation tests** for validating experimental results
- **Performance tests** for benchmarking optimizations

### Test Structure
```python
import pytest
from src.agents.resource_agent import ResourceAgent

class TestResourceAgent:
    def test_job_scoring_algorithm(self):
        # Arrange
        agent = ResourceAgent(...)
        job_data = {...}
        
        # Act
        score = agent._calculate_job_score(job_data)
        
        # Assert
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
```

### Evaluation Test Requirements

For new evaluation scenarios:

1. **Reproducible results** with fixed random seeds
2. **Statistical validation** with multiple repetitions
3. **Performance benchmarks** with timing measurements
4. **Clear documentation** of experimental setup
5. **Comparison baselines** against existing methods

## 📊 Adding New Evaluation Scenarios

### Template for New Scenarios

```python
from evaluation.systematic_resilience_evaluation import ExperimentConfig

def create_custom_scenario():
    """
    Create custom evaluation scenario.
    
    Returns:
        ExperimentConfig: Configuration for the new scenario
    """
    return ExperimentConfig(
        name="CustomScenario-Description",
        num_jobs=100,
        num_agents=10,
        agent_failure_rate=0.15,
        scheduler_failure_rate=0.05,
        job_arrival_pattern='custom',
        failure_pattern='custom',
        simulation_time=120.0,
        repetitions=3
    )

def run_custom_evaluation():
    """Run custom evaluation with proper documentation."""
    config = create_custom_scenario()
    results = run_resilience_experiment(config)
    
    # Document results
    print(f"Custom scenario results: {results}")
    return results
```

### Documentation Requirements

For new scenarios, include:

1. **Purpose**: What does this scenario test?
2. **Methodology**: How is it implemented?
3. **Expected results**: What outcomes are anticipated?
4. **Validation**: How are results verified?
5. **Comparison**: How does it relate to existing scenarios?

## 🐛 Bug Reports

### Issue Template

When reporting bugs, include:

**Bug Description**
- Clear description of the issue
- Expected vs. actual behavior

**Reproduction Steps**
1. Step-by-step instructions
2. Code snippets if applicable
3. Configuration details

**Environment**
- Python version
- Operating system
- Package versions (`pip freeze`)

**Additional Context**
- Error messages and stack traces
- Screenshots if relevant
- Related issues or discussions

## 💡 Feature Requests

### Enhancement Template

**Feature Description**
- Clear description of the proposed feature
- Use cases and motivation
- Potential impact on existing functionality

**Implementation Ideas**
- Suggested approach (if any)
- Alternative solutions considered
- Resources or references

**Acceptance Criteria**
- How to verify the feature works
- Performance requirements
- Documentation needs

## 📚 Documentation Standards

### Code Documentation

```python
def calculate_resilience_score(
    completion_rate: float,
    agent_failures: int,
    scheduler_failures: int
) -> float:
    """
    Calculate composite resilience score for scheduling system.
    
    Args:
        completion_rate: Job completion rate (0.0-1.0)
        agent_failures: Number of agent failures observed
        scheduler_failures: Number of scheduler failures observed
    
    Returns:
        float: Resilience score (0-100), higher is better
        
    Raises:
        ValueError: If completion_rate is not in [0.0, 1.0]
        
    Example:
        >>> score = calculate_resilience_score(0.95, 2, 0)
        >>> print(f"Resilience score: {score}")
        Resilience score: 87.5
    """
```

### README Updates

When adding features:

1. Update the **Features** section
2. Add **Usage examples**
3. Update **Installation** if needed
4. Add **Performance benchmarks** for significant changes

## 🔄 Release Process

### Versioning

We use semantic versioning (SemVer):

- **Major** (1.0.0): Breaking changes
- **Minor** (1.1.0): New features, backward compatible
- **Patch** (1.1.1): Bug fixes, backward compatible

### Release Checklist

Before releasing:

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number incremented
- [ ] CHANGELOG.md updated
- [ ] Performance benchmarks run
- [ ] Evaluation results validated

## 👥 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be respectful** in all interactions
- **Be collaborative** and help others learn
- **Be patient** with questions and feedback
- **Be constructive** in criticism and suggestions

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions and reviews
- **Email**: Direct contact for sensitive issues

## 🏆 Recognition

Contributors will be:

- **Listed** in the contributors section
- **Credited** in release notes
- **Acknowledged** in academic publications (with permission)
- **Invited** to co-author follow-up research (for significant contributions)

## ❓ Questions?

- Check the [FAQ](docs/FAQ.md)
- Search [existing issues](https://github.com/username/distributed-multiagent-scheduling/issues)
- Start a [discussion](https://github.com/username/distributed-multiagent-scheduling/discussions)
- Contact the maintainers: [authors@institution.edu](mailto:authors@institution.edu)

Thank you for contributing to distributed multi-agent scheduling research! 🚀
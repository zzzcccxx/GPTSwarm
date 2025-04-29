# Shapley Value Optimization for GPTSwarm

This extension to GPTSwarm adds a secondary optimization step based on cooperative game theory's Shapley values to better refine the network topology of multi-agent systems.

## Overview

The system works in two steps:

1. **Initial Optimization**: Gradient-based optimization of edge connections (original approach)
2. **Shapley Value Optimization**: Secondary optimization that evaluates the marginal contribution of each potential edge to the overall system performance

This approach helps identify which connections between agents are most valuable for solving a specific task, leading to more efficient and effective multi-agent topologies.

## How It Works

Shapley values measure each edge's contribution by:

1. Creating random permutations of edges
2. For each edge, calculating its marginal contribution when added to a subset of other edges
3. Averaging these marginal contributions across many samples to get a robust estimate

Edges with high Shapley values contribute significantly to performance, while those with low or negative values may be detrimental.

## Usage

To use Shapley value optimization, run the following command:

```bash
python experiments/run_mmlu_multi_agent.py --use_shapley --visualize_shapley
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_shapley` | Enable Shapley value optimization | False |
| `--shapley_samples` | Number of Monte Carlo samples for Shapley estimation | 20 |
| `--shapley_threshold` | Threshold for including edges based on Shapley values | 0.0 |
| `--shapley_lr` | Learning rate for Shapley updates | 0.2 |
| `--visualize_shapley` | Generate Shapley value visualizations | False |
| `--shapley_max_edges` | Maximum number of edges to evaluate for Shapley values | 50 |
| `--shapley_time_budget` | Time budget for Shapley computation in seconds | 600 |
| `--shapley_parallel` | Use parallel computation for Shapley values | True |
| `--shapley_batch_size` | Batch size for parallel Shapley computation | 5 |

## Performance Optimization

Shapley value computation can be computationally expensive. To improve performance:

1. **Reduce sample size**: Lower the `--shapley_samples` value (e.g., 5-10) for faster computation at the cost of some accuracy
2. **Limit evaluated edges**: Use `--shapley_max_edges` to focus on the most promising connections
3. **Set a time budget**: Use `--shapley_time_budget` to limit total computation time
4. **Parallel processing**: Enable `--shapley_parallel` to compute values concurrently
5. **Adjust batch size**: Use `--shapley_batch_size` to control the number of concurrent evaluations

For quick experiments, consider:

```bash
python experiments/run_mmlu_multi_agent.py --use_shapley --shapley_samples 5 --shapley_max_edges 20 --shapley_time_budget 300
```

## Visualization

With the `--visualize_shapley` flag, the system generates several visualizations in the `runs/<timestamp>/shapley_analysis/` directory:

- **Shapley value distribution**: Histogram of Shapley values across all edges
- **Shapley values vs. initial probabilities**: Scatter plot showing relationship between initial edge probabilities and Shapley values
- **Top edges by Shapley value**: Bar chart of the most important connections
- **Edge report (CSV)**: Detailed report on each edge with various metrics
- **Topology comparison**: Statistics comparing original and Shapley-optimized topologies

## Example

Here's a typical workflow:

1. Run initial gradient-based optimization
2. Apply Shapley value optimization to identify key connections
3. Generate a new graph topology based on Shapley values
4. Evaluate the new topology against the original one

## Benefits

- More robust identification of important connections
- Better elimination of detrimental connections
- Insights into agent interaction patterns
- Improved overall system performance

## Implementation Details

The implementation consists of:

- `ShapleyEdgeOptimizer`: Computes Shapley values and optimizes connections
- `ShapleyVisualizer`: Generates visualizations and reports
- Integration into the existing `Evaluator` class

The optimizer uses Monte Carlo sampling to approximate Shapley values, which is more computationally efficient than computing exact values for large graphs.

## References

- Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games, 2(28), 307-317.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30. 
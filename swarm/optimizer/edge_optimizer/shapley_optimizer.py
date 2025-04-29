#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import asyncio
from typing import List, Dict, Tuple, Set, Any, Optional, Callable
from copy import deepcopy
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

from swarm.graph.composite_graph import CompositeGraph
from swarm.optimizer.edge_optimizer.parameterization import EdgeWiseDistribution


class ShapleyEdgeOptimizer:
    """
    Secondary optimization step using Shapley values to refine edge connection structure.
    
    This optimizer takes the results from the initial EdgeWiseDistribution optimization
    and applies cooperative game theory principles to better evaluate the contribution
    of each potential edge to the overall performance.
    """
    
    def __init__(
            self, 
            connection_dist: EdgeWiseDistribution,
            composite_graph: CompositeGraph,
            num_samples: int = 100,
            threshold: float = 0.5,
            max_edges_to_evaluate: int = 10,  # 最多评估多少条边
            time_budget_seconds: int = 300,   # 时间预算（秒）
            use_parallel: bool = True,        # 是否使用并行计算
            batch_size: int = 5               # 批处理大小
    ):
        """
        Initialize the Shapley value edge optimizer.
        
        Args:
            connection_dist: The trained EdgeWiseDistribution with edge probabilities
            composite_graph: The composite graph structure
            num_samples: Number of random samples to estimate Shapley values
            threshold: Probability threshold for initial edge inclusion
            max_edges_to_evaluate: Maximum number of edges to evaluate (0 for all)
            time_budget_seconds: Maximum computation time in seconds
            use_parallel: Whether to use parallel computation
            batch_size: Size of edge batches for evaluation
        """
        self.connection_dist = connection_dist
        self.composite_graph = composite_graph
        self.num_samples = num_samples
        self.threshold = threshold
        self.edge_shapley_values = None
        self.edge_initial_probs = torch.sigmoid(self.connection_dist.edge_logits)
        self.max_edges_to_evaluate = max_edges_to_evaluate
        self.time_budget_seconds = time_budget_seconds
        self.use_parallel = use_parallel
        self.batch_size = batch_size
        
    async def evaluate_graph(self, graph: CompositeGraph, eval_fn) -> float:
        """
        Evaluate a graph configuration using the provided evaluation function.
        
        Args:
            graph: The graph configuration to evaluate
            eval_fn: Async function that takes a graph and returns a performance score
            
        Returns:
            Performance score for the given graph
        """
        return await eval_fn(graph)
    
    async def _compute_edge_shapley(self, edge_idx: int, eval_fn) -> float:
        """
        Compute the Shapley value for a single edge.
        
        Args:
            edge_idx: Index of the edge
            eval_fn: Function to evaluate graph performance
            
        Returns:
            Shapley value for the edge
        """
        num_edges = len(self.connection_dist.potential_connections)
        marginal_contributions = []
        
        for _ in range(self.num_samples):
            # Create random permutation of edges
            perm = torch.randperm(num_edges)
            
            # Get position of current edge in permutation
            pos = torch.where(perm == edge_idx)[0].item()
            
            # Create graph with edges before current edge in permutation
            before_mask = torch.zeros(num_edges, dtype=torch.bool)
            for i in range(pos):
                before_mask[perm[i]] = self.edge_initial_probs[perm[i]] > self.threshold
            graph_without = self.connection_dist.realize_mask(self.composite_graph, before_mask)
            
            # Create graph with edges before plus current edge
            after_mask = before_mask.clone()
            after_mask[edge_idx] = True
            graph_with = self.connection_dist.realize_mask(self.composite_graph, after_mask)
            
            # Compute marginal contribution
            score_with = await self.evaluate_graph(graph_with, eval_fn)
            score_without = await self.evaluate_graph(graph_without, eval_fn)
            marginal_contribution = score_with - score_without
            marginal_contributions.append(marginal_contribution)
        
        # Average the marginal contributions to get Shapley value
        return np.mean(marginal_contributions)
    
    async def compute_shapley_values(self, eval_fn) -> torch.Tensor:
        """
        Compute Shapley values for all potential edges using Monte Carlo sampling.
        
        Args:
            eval_fn: Async function that evaluates a graph configuration
            
        Returns:
            Tensor of Shapley values for each potential edge
        """
        start_time = time.time()
        num_edges = len(self.connection_dist.potential_connections)
        
        # Determine edges to evaluate
        if self.max_edges_to_evaluate > 0 and self.max_edges_to_evaluate < num_edges:
            # Use edges with highest initial probabilities
            probs_with_indices = [(prob.item(), i) for i, prob in enumerate(self.edge_initial_probs)]
            probs_with_indices.sort(reverse=True)
            
            # Take top edges by probability and some random edges
            top_edges_count = int(self.max_edges_to_evaluate * 0.8)
            random_edges_count = self.max_edges_to_evaluate - top_edges_count
            
            # Get top edges
            edges_to_evaluate = [idx for _, idx in probs_with_indices[:top_edges_count]]
            
            # Add some random edges not in top edges
            potential_random = [idx for _, idx in probs_with_indices[top_edges_count:]]
            if potential_random and random_edges_count > 0:
                random_indices = np.random.choice(
                    len(potential_random), 
                    size=min(random_edges_count, len(potential_random)), 
                    replace=False
                )
                edges_to_evaluate.extend([potential_random[i] for i in random_indices])
        else:
            edges_to_evaluate = list(range(num_edges))
        
        print(f"Evaluating Shapley values for {len(edges_to_evaluate)} edges out of {num_edges}")
        
        # Initialize result tensor
        shapley_values = torch.zeros(num_edges)
        
        # Use default value for edges we don't evaluate
        # We use a small positive value to not completely exclude them
        default_shapley = 0.01
        for i in range(num_edges):
            if i not in edges_to_evaluate:
                shapley_values[i] = default_shapley
        
        # Track progress
        evaluated_edges = 0
        
        if self.use_parallel:
            # Process in batches to avoid too many concurrent evaluations
            for batch_start in range(0, len(edges_to_evaluate), self.batch_size):
                # Check time budget
                if time.time() - start_time > self.time_budget_seconds:
                    print(f"Time budget exceeded after evaluating {evaluated_edges} edges")
                    break
                
                # Get current batch
                batch_end = min(batch_start + self.batch_size, len(edges_to_evaluate))
                current_batch = edges_to_evaluate[batch_start:batch_end]
                
                # Compute Shapley values in parallel
                tasks = [self._compute_edge_shapley(edge_idx, eval_fn) for edge_idx in current_batch]
                batch_results = await asyncio.gather(*tasks)
                
                # Update results
                for edge_idx, value in zip(current_batch, batch_results):
                    shapley_values[edge_idx] = value
                
                evaluated_edges += len(current_batch)
                print(f"Evaluated {evaluated_edges}/{len(edges_to_evaluate)} edges, "
                      f"elapsed time: {time.time() - start_time:.1f}s")
        else:
            # Sequential evaluation
            for edge_idx in edges_to_evaluate:
                # Check time budget
                if time.time() - start_time > self.time_budget_seconds:
                    print(f"Time budget exceeded after evaluating {evaluated_edges} edges")
                    break
                
                value = await self._compute_edge_shapley(edge_idx, eval_fn)
                shapley_values[edge_idx] = value
                
                evaluated_edges += 1
                if evaluated_edges % 5 == 0:
                    print(f"Evaluated {evaluated_edges}/{len(edges_to_evaluate)} edges, "
                          f"elapsed time: {time.time() - start_time:.1f}s")
        
        print(f"Shapley evaluation completed in {time.time() - start_time:.1f} seconds")
        
        self.edge_shapley_values = shapley_values
        return shapley_values
    
    def optimize_connections(self, shapley_values: Optional[torch.Tensor] = None, shapley_threshold: float = 0.0) -> Tuple[CompositeGraph, torch.Tensor]:
        """
        Create an optimized graph based on Shapley values.
        
        Args:
            shapley_values: Pre-computed Shapley values (if None, use computed values)
            shapley_threshold: Threshold for Shapley values to include an edge
            
        Returns:
            Tuple of (optimized graph, edge mask)
        """
        if shapley_values is None:
            shapley_values = self.edge_shapley_values
            
        if shapley_values is None:
            raise ValueError("Shapley values must be computed before optimization")
        
        # Create edge mask based on Shapley values
        shapley_mask = shapley_values > shapley_threshold
        
        # Create optimized graph
        optimized_graph = self.connection_dist.realize_mask(
            self.composite_graph, 
            shapley_mask
        )
        
        return optimized_graph, shapley_mask
    
    def update_connection_dist(self, shapley_values: torch.Tensor, learning_rate: float = 0.1) -> None:
        """
        Update the edge logits based on Shapley values.
        
        Args:
            shapley_values: Computed Shapley values
            learning_rate: Learning rate for updating edge logits
        """
        # Normalize Shapley values to [0, 1] range for better comparison
        normalized_shapley = (shapley_values - shapley_values.min()) / (shapley_values.max() - shapley_values.min() + 1e-8)
        
        # Update edge logits based on Shapley values
        with torch.no_grad():
            # Convert normalized Shapley values to logits
            shapley_logits = torch.log(normalized_shapley / (1 - normalized_shapley + 1e-8) + 1e-8)
            
            # Update edge logits with a weighted combination
            self.connection_dist.edge_logits.data = (1 - learning_rate) * self.connection_dist.edge_logits.data + learning_rate * shapley_logits 
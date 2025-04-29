#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from typing import List, Dict, Optional, Tuple

from swarm.optimizer.edge_optimizer.parameterization import EdgeWiseDistribution
from swarm.graph.composite_graph import CompositeGraph


class ShapleyVisualizer:
    """
    Visualization tools for analyzing Shapley values and edge probabilities in a swarm graph.
    """
    
    def __init__(
            self, 
            connection_dist: EdgeWiseDistribution,
            composite_graph: CompositeGraph,
            save_dir: Optional[str] = None
    ):
        """
        Initialize the Shapley value visualizer.
        
        Args:
            connection_dist: The trained EdgeWiseDistribution with edge probabilities
            composite_graph: The composite graph structure
            save_dir: Directory to save visualization outputs
        """
        self.connection_dist = connection_dist
        self.composite_graph = composite_graph
        self.save_dir = save_dir
        
        # Create save directory if not exists
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
    
    def plot_shapley_distribution(self, shapley_values: torch.Tensor, title: str = "Shapley Value Distribution") -> None:
        """
        Plot the distribution of Shapley values.
        
        Args:
            shapley_values: Tensor of computed Shapley values
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(shapley_values.numpy(), kde=True)
        plt.title(title)
        plt.xlabel("Shapley Value")
        plt.ylabel("Frequency")
        
        if self.save_dir is not None:
            plt.savefig(os.path.join(self.save_dir, "shapley_distribution.png"))
        
        plt.close()
    
    def plot_shapley_vs_initial_probs(
            self, 
            shapley_values: torch.Tensor, 
            normalized: bool = True
    ) -> None:
        """
        Plot Shapley values against initial edge probabilities.
        
        Args:
            shapley_values: Tensor of computed Shapley values
            normalized: Whether to normalize Shapley values to [0,1]
        """
        initial_probs = torch.sigmoid(self.connection_dist.edge_logits).detach().numpy()
        
        # Normalize Shapley values if needed
        if normalized:
            shapley_norm = (shapley_values - shapley_values.min()) / (shapley_values.max() - shapley_values.min() + 1e-8)
            shapley_np = shapley_norm.numpy()
            ylabel = "Normalized Shapley Value"
        else:
            shapley_np = shapley_values.numpy()
            ylabel = "Shapley Value"
            
        plt.figure(figsize=(10, 6))
        plt.scatter(initial_probs, shapley_np, alpha=0.6)
        
        # Add regression line
        z = np.polyfit(initial_probs, shapley_np, 1)
        p = np.poly1d(z)
        plt.plot(initial_probs, p(initial_probs), "r--", alpha=0.8)
        
        plt.title("Shapley Values vs Initial Edge Probabilities")
        plt.xlabel("Initial Edge Probability")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        
        if self.save_dir is not None:
            plt.savefig(os.path.join(self.save_dir, "shapley_vs_initial_probs.png"))
        
        plt.close()
    
    def plot_edge_rankings(
            self, 
            shapley_values: torch.Tensor,
            top_n: int = 20,
            annotate_nodes: bool = True
    ) -> None:
        """
        Plot the top edges by Shapley value with their node names.
        
        Args:
            shapley_values: Tensor of computed Shapley values
            top_n: Number of top edges to display
            annotate_nodes: Whether to annotate node names
        """
        # Get sorted indices by Shapley value
        sorted_indices = torch.argsort(shapley_values, descending=True)
        
        # Take top N
        top_indices = sorted_indices[:top_n]
        
        # Create data for plotting
        top_shapley = shapley_values[top_indices].numpy()
        
        # Get edge labels
        edge_labels = []
        for idx in top_indices:
            src_id, dst_id = self.connection_dist.potential_connections[idx]
            src_node = self.composite_graph.find_node(src_id)
            dst_node = self.composite_graph.find_node(dst_id)
            edge_labels.append(f"{src_node.node_name} → {dst_node.node_name}")
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_shapley)), top_shapley, color='skyblue')
        plt.yticks(range(len(top_shapley)), edge_labels)
        plt.xlabel('Shapley Value')
        plt.title(f'Top {top_n} Edges by Shapley Value')
        plt.tight_layout()
        
        if self.save_dir is not None:
            plt.savefig(os.path.join(self.save_dir, "top_edges_by_shapley.png"))
        
        plt.close()
    
    def generate_edge_report(
            self, 
            shapley_values: torch.Tensor, 
            threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Generate a detailed report of edges with their Shapley values and probabilities.
        
        Args:
            shapley_values: Tensor of computed Shapley values
            threshold: Threshold for including edges in the final graph
            
        Returns:
            DataFrame with edge details
        """
        # Normalize Shapley values to [0,1]
        shapley_norm = (shapley_values - shapley_values.min()) / (shapley_values.max() - shapley_values.min() + 1e-8)
        
        # Get initial probabilities
        initial_probs = torch.sigmoid(self.connection_dist.edge_logits).detach()
        
        # Create data frame
        data = []
        for i, (conn, shapley, shapley_n, init_prob) in enumerate(zip(
                self.connection_dist.potential_connections, 
                shapley_values, 
                shapley_norm,
                initial_probs)):
            
            src_id, dst_id = conn
            src_node = self.composite_graph.find_node(src_id)
            dst_node = self.composite_graph.find_node(dst_id)
            
            data.append({
                'edge_idx': i,
                'source_node': src_node.node_name,
                'source_id': src_id,
                'target_node': dst_node.node_name,
                'target_id': dst_id,
                'initial_probability': init_prob.item(),
                'shapley_value': shapley.item(),
                'normalized_shapley': shapley_n.item(),
                'include_by_initial_prob': init_prob.item() > threshold,
                'include_by_shapley': shapley.item() > threshold
            })
        
        df = pd.DataFrame(data)
        
        # Sort by Shapley value
        df = df.sort_values('shapley_value', ascending=False)
        
        # Save to CSV if directory is provided
        if self.save_dir is not None:
            df.to_csv(os.path.join(self.save_dir, "edge_report.csv"), index=False)
        
        return df
    
    def compare_topologies(
            self, 
            shapley_values: torch.Tensor,
            threshold: float = 0.0,
            initial_threshold: float = 0.5
    ) -> Dict[str, int]:
        """
        Compare the topologies based on initial probabilities vs Shapley values.
        
        Args:
            shapley_values: Tensor of computed Shapley values
            threshold: Threshold for Shapley values
            initial_threshold: Threshold for initial probabilities
            
        Returns:
            Dictionary with topology comparison statistics
        """
        initial_mask = torch.sigmoid(self.connection_dist.edge_logits) > initial_threshold
        shapley_mask = shapley_values > threshold
        
        # Calculate statistics
        total_edges = len(self.connection_dist.potential_connections)
        initial_edges = initial_mask.sum().item()
        shapley_edges = shapley_mask.sum().item()
        
        # Common edges
        common_edges = (initial_mask & shapley_mask).sum().item()
        
        # Unique edges in each
        only_initial = (initial_mask & ~shapley_mask).sum().item()
        only_shapley = (~initial_mask & shapley_mask).sum().item()
        
        # Calculate percentages
        common_percent = (common_edges / total_edges) * 100
        different_percent = ((only_initial + only_shapley) / total_edges) * 100
        
        stats = {
            'total_potential_edges': total_edges,
            'edges_by_initial_prob': initial_edges,
            'edges_by_shapley': shapley_edges,
            'common_edges': common_edges,
            'only_in_initial': only_initial,
            'only_in_shapley': only_shapley,
            'common_percentage': common_percent,
            'different_percentage': different_percent
        }
        
        # Save to txt if directory is provided
        if self.save_dir is not None:
            with open(os.path.join(self.save_dir, "topology_comparison.txt"), 'w') as f:
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
        
        return stats 
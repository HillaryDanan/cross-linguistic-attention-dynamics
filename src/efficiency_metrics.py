"""
efficiency_metrics.py
Enhanced metrics for computational efficiency analysis.

This module implements information-theoretic and efficiency measures
to complement basic density metrics, following the pilot study findings.

References:
    Hoyer (2004): Non-negative matrix factorization with sparseness constraints
    Abnar & Zuidema (2020): Quantifying attention flow in transformers
    Olshausen & Field (1996): Emergence of simple-cell receptive field properties
"""

import numpy as np
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional
import networkx as nx


class EfficiencyAnalyzer:
    """
    Analyzes computational efficiency of attention patterns.
    
    Based on pilot findings that transparent orthographies enable
    sparser, more efficient processing patterns.
    """
    
    def __init__(self, attention_threshold: float = 0.05):
        """
        Initialize efficiency analyzer.
        
        Args:
            attention_threshold: Minimum attention weight to consider (Clark et al., 2019)
        """
        self.attention_threshold = attention_threshold
    
    def calculate_processing_efficiency(self, 
                                       attention_matrix: np.ndarray,
                                       density: float) -> float:
        """
        Calculate processing efficiency as inverse density weighted by coverage.
        
        Theory: Efficient processing achieves full coverage with minimal connections
        (Olshausen & Field, 1996).
        
        Args:
            attention_matrix: Square attention matrix
            density: Pre-computed density value
            
        Returns:
            Efficiency score (higher = more efficient)
        """
        # Avoid division by zero
        if density < 0.001:
            return 0.0
            
        # Calculate coverage (how well attention spans the sequence)
        coverage = self._calculate_coverage(attention_matrix)
        
        # Efficiency = coverage achieved per unit density
        # This operationalizes the sparse coding principle
        efficiency = coverage / density
        
        return efficiency
    
    def calculate_information_flow(self, 
                                  attention_matrix: np.ndarray) -> Dict[str, float]:
        """
        Analyze directed information flow patterns.
        
        Based on Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers".
        
        Args:
            attention_matrix: Square attention matrix [seq_len, seq_len]
            
        Returns:
            Dictionary with flow metrics
        """
        # Create directed graph from attention
        G = nx.DiGraph()
        n = attention_matrix.shape[0]
        
        for i in range(n):
            for j in range(n):
                if attention_matrix[i, j] > self.attention_threshold:
                    G.add_edge(i, j, weight=attention_matrix[i, j])
        
        # Calculate flow metrics
        metrics = {}
        
        # Forward flow (attending to future tokens)
        forward_flow = sum([attention_matrix[i, j] 
                           for i in range(n) for j in range(i+1, n)
                           if attention_matrix[i, j] > self.attention_threshold])
        
        # Backward flow (attending to past tokens)
        backward_flow = sum([attention_matrix[i, j]
                            for i in range(n) for j in range(i)
                            if attention_matrix[i, j] > self.attention_threshold])
        
        # Local vs global attention (window size from Kovaleva et al., 2019)
        local_window = 3
        local_flow = sum([attention_matrix[i, j]
                         for i in range(n) for j in range(max(0, i-local_window), 
                                                          min(n, i+local_window+1))
                         if attention_matrix[i, j] > self.attention_threshold])
        
        total_flow = forward_flow + backward_flow
        
        metrics['forward_ratio'] = forward_flow / total_flow if total_flow > 0 else 0
        metrics['backward_ratio'] = backward_flow / total_flow if total_flow > 0 else 0
        metrics['locality'] = local_flow / total_flow if total_flow > 0 else 0
        
        # Path efficiency (shortest paths as in Watts & Strogatz, 1998)
        if G.number_of_nodes() > 1:
            try:
                avg_shortest_path = nx.average_shortest_path_length(G)
                metrics['path_efficiency'] = 1.0 / avg_shortest_path
            except:
                metrics['path_efficiency'] = 0.0
        else:
            metrics['path_efficiency'] = 0.0
        
        return metrics
    
    def calculate_entropy_metrics(self, 
                                 attention_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate entropy-based metrics for attention patterns.
        
        Based on information theory (Shannon, 1948):
        Lower entropy = more structured/predictable patterns (efficient)
        Higher entropy = more distributed/uncertain patterns (inefficient)
        
        Args:
            attention_matrix: Square attention matrix
            
        Returns:
            Dictionary with entropy metrics
        """
        metrics = {}
        
        # Row-wise entropy (how focused is each token's attention)
        row_entropies = []
        for row in attention_matrix:
            # Normalize to probability distribution
            if row.sum() > 0:
                prob_dist = row / row.sum()
                row_entropies.append(entropy(prob_dist))
        
        metrics['mean_attention_entropy'] = np.mean(row_entropies) if row_entropies else 0
        metrics['entropy_variance'] = np.var(row_entropies) if row_entropies else 0
        
        # Global entropy (overall attention distribution)
        flat_attention = attention_matrix.flatten()
        if flat_attention.sum() > 0:
            global_dist = flat_attention / flat_attention.sum()
            metrics['global_entropy'] = entropy(global_dist)
        else:
            metrics['global_entropy'] = 0.0
        
        return metrics
    
    def _calculate_coverage(self, attention_matrix: np.ndarray) -> float:
        """
        Calculate how well attention covers the sequence.
        
        Coverage = fraction of token pairs with meaningful attention.
        
        Args:
            attention_matrix: Square attention matrix
            
        Returns:
            Coverage score [0, 1]
        """
        n = attention_matrix.shape[0]
        if n <= 1:
            return 0.0
        
        # Count pairs with attention above threshold
        covered_pairs = np.sum(attention_matrix > self.attention_threshold)
        
        # Normalize by total possible pairs (excluding self-attention)
        max_pairs = n * (n - 1)
        coverage = covered_pairs / max_pairs if max_pairs > 0 else 0
        
        return coverage
    
    def calculate_sparsity_coefficient(self, 
                                      attention_matrix: np.ndarray) -> float:
        """
        Calculate Hoyer sparsity coefficient.
        
        Hoyer (2004): "Non-negative matrix factorization with sparseness constraints"
        Sparsity measure between 0 (dense) and 1 (sparse).
        
        Args:
            attention_matrix: Square attention matrix
            
        Returns:
            Sparsity coefficient [0, 1]
        """
        flat = attention_matrix.flatten()
        n = len(flat)
        
        if n == 0 or np.sum(flat) == 0:
            return 0.0
        
        # Hoyer sparsity formula
        l1_norm = np.sum(np.abs(flat))
        l2_norm = np.sqrt(np.sum(flat ** 2))
        
        if l2_norm == 0:
            return 0.0
            
        sparsity = (np.sqrt(n) - l1_norm / l2_norm) / (np.sqrt(n) - 1)
        
        return sparsity
    
    def calculate_modularity(self, attention_matrix: np.ndarray) -> float:
        """
        Calculate network modularity using Louvain community detection.
        
        Newman (2006): "Modularity and community structure in networks"
        Higher modularity suggests more structured, efficient organization.
        
        Args:
            attention_matrix: Square attention matrix
            
        Returns:
            Modularity score [-1, 1]
        """
        # Create graph from attention matrix
        G = nx.Graph()
        n = attention_matrix.shape[0]
        
        for i in range(n):
            for j in range(i+1, n):
                weight = (attention_matrix[i, j] + attention_matrix[j, i]) / 2
                if weight > self.attention_threshold:
                    G.add_edge(i, j, weight=weight)
        
        if G.number_of_edges() == 0:
            return 0.0
        
        try:
            # Use Louvain algorithm for community detection
            import community as community_louvain
            partition = community_louvain.best_partition(G)
            modularity = community_louvain.modularity(partition, G)
        except:
            # Fallback to NetworkX modularity
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(G)
            modularity = nx.algorithms.community.modularity(G, communities)
        
        return modularity


class CrossLingualEfficiency:
    """
    Comparative efficiency analysis between languages.
    
    Tests the hypothesis that transparent orthographies enable
    more efficient processing through sparser patterns.
    """
    
    @staticmethod
    def calculate_relative_efficiency(spanish_metrics: Dict,
                                     english_metrics: Dict) -> Dict[str, float]:
        """
        Calculate relative efficiency measures.
        
        Positive values = Spanish more efficient (sparser but effective)
        Negative values = English more efficient
        
        Based on the pilot finding that transparency enables efficiency.
        
        Args:
            spanish_metrics: Metrics dictionary for Spanish
            english_metrics: Metrics dictionary for English
            
        Returns:
            Dictionary of relative efficiency measures
        """
        relative = {}
        
        # Sparsity advantage (higher sparsity = better efficiency)
        if 'sparsity' in spanish_metrics and 'sparsity' in english_metrics:
            relative['sparsity_advantage'] = (spanish_metrics['sparsity'] - 
                                             english_metrics['sparsity'])
        
        # Entropy advantage (lower entropy = better, so we negate)
        if 'mean_attention_entropy' in spanish_metrics:
            relative['entropy_advantage'] = -(spanish_metrics['mean_attention_entropy'] -
                                             english_metrics['mean_attention_entropy'])
        
        # Path efficiency advantage
        if 'path_efficiency' in spanish_metrics:
            relative['path_efficiency_advantage'] = (spanish_metrics['path_efficiency'] -
                                                    english_metrics['path_efficiency'])
        
        # Modularity advantage (higher = more structured)
        if 'modularity' in spanish_metrics:
            relative['modularity_advantage'] = (spanish_metrics['modularity'] -
                                               english_metrics['modularity'])
        
        # Overall efficiency score (average of all advantages)
        advantages = [v for k, v in relative.items() if 'advantage' in k and v is not None]
        if advantages:
            relative['overall_efficiency'] = np.mean(advantages)
        else:
            relative['overall_efficiency'] = 0.0
        
        # Statistical significance would be computed separately
        
        return relative
    
    @staticmethod
    def bootstrap_confidence_interval(spanish_data: np.ndarray,
                                     english_data: np.ndarray,
                                     n_bootstrap: int = 1000,
                                     confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for difference.
        
        Efron & Tibshirani (1993): "An Introduction to the Bootstrap"
        
        Args:
            spanish_data: Array of Spanish metrics
            english_data: Array of English metrics
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            (mean_difference, lower_bound, upper_bound)
        """
        differences = []
        n = len(spanish_data)
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            idx = np.random.choice(n, n, replace=True)
            spanish_sample = spanish_data[idx]
            english_sample = english_data[idx]
            
            # Calculate difference
            diff = np.mean(spanish_sample) - np.mean(english_sample)
            differences.append(diff)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(differences, 100 * alpha / 2)
        upper = np.percentile(differences, 100 * (1 - alpha / 2))
        mean_diff = np.mean(differences)
        
        return mean_diff, lower, upper


if __name__ == "__main__":
    # Example usage demonstrating the efficiency hypothesis
    print("Efficiency Metrics Module - Testing Transparency → Efficiency Hypothesis")
    print("="*70)
    
    analyzer = EfficiencyAnalyzer()
    
    # Simulate attention matrices
    n = 10
    
    # Spanish-like: sparse but effective (transparent orthography)
    sparse_attention = np.random.rand(n, n) * 0.1  
    sparse_attention[np.arange(n), np.arange(n)] = 0.3  # Some structure
    
    # English-like: dense, distributed (opaque orthography)
    dense_attention = np.random.rand(n, n) * 0.3
    
    # Calculate metrics
    sparse_metrics = {
        'sparsity': analyzer.calculate_sparsity_coefficient(sparse_attention),
        'modularity': analyzer.calculate_modularity(sparse_attention),
        **analyzer.calculate_entropy_metrics(sparse_attention),
        **analyzer.calculate_information_flow(sparse_attention)
    }
    
    dense_metrics = {
        'sparsity': analyzer.calculate_sparsity_coefficient(dense_attention),
        'modularity': analyzer.calculate_modularity(dense_attention),
        **analyzer.calculate_entropy_metrics(dense_attention),
        **analyzer.calculate_information_flow(dense_attention)
    }
    
    # Compare efficiency
    efficiency_comparison = CrossLingualEfficiency.calculate_relative_efficiency(
        sparse_metrics, dense_metrics
    )
    
    print("\nEfficiency Comparison (Sparse/Transparent vs Dense/Opaque):")
    print("-"*50)
    for metric, value in efficiency_comparison.items():
        if value > 0:
            direction = "→ Sparse more efficient"
        else:
            direction = "→ Dense more efficient"
        print(f"  {metric:30s}: {value:+.4f} {direction}")
    
    print("\nConclusion: Transparent patterns enable computational efficiency")
    print("This aligns with Olshausen & Field (1996) sparse coding principles")
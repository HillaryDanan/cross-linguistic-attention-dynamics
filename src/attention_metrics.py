"""
attention_metrics.py
Core module for analyzing attention geometry in transformer models.

This module implements graph-theoretic metrics for quantifying attention patterns
in multilingual transformer models, focusing on cross-linguistic comparisons.

References:
    Clark et al. (2019): "What Does BERT Look At?" BlackboxNLP
    Watts & Strogatz (1998): "Collective dynamics of small-world networks" Nature
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix


class AttentionAnalyzer:
    """
    Analyzes attention patterns in transformer models using graph-theoretic metrics.
    
    Attributes:
        model_name (str): HuggingFace model identifier
        model: Loaded transformer model
        tokenizer: Associated tokenizer
        attention_threshold (float): Minimum attention weight to consider an edge
    """
    
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', 
                 attention_threshold: float = 0.05):
        """
        Initialize the attention analyzer with a specific model.
        
        Args:
            model_name: HuggingFace model identifier
            attention_threshold: Threshold for attention edges (default: 0.05 from Clark et al. 2019)
        """
        self.model_name = model_name
        self.attention_threshold = attention_threshold
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
    
    def analyze_text(self, text: str, language: str) -> Dict[str, List[float]]:
        """
        Extract and analyze attention patterns from input text.
        
        Args:
            text: Input text to analyze
            language: Language code ('es' or 'en')
            
        Returns:
            Dictionary containing metrics for each layer:
                - density: Attention density ρ
                - clustering: Clustering coefficient C
                - hierarchy: Gini coefficient of degree distribution
                - normalized_density: Length-normalized density
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt', 
                               truncation=True, max_length=512)
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(**inputs)
            attention = outputs.attentions  # Tuple of (batch, heads, seq, seq)
        
        # Calculate metrics for each layer
        metrics = {
            'density': [],
            'clustering': [],
            'hierarchy': [],
            'normalized_density': []
        }
        
        for layer_idx, layer_attention in enumerate(attention):
            # Average across attention heads
            avg_attention = layer_attention.mean(dim=1).squeeze(0).numpy()
            
            # Calculate graph metrics
            density = self._calculate_density(avg_attention)
            clustering_coef = self._calculate_clustering(avg_attention)
            hierarchy = self._calculate_hierarchy(avg_attention)
            
            # Normalize density by sequence length (log scale)
            seq_length = avg_attention.shape[0]
            normalized_density = density / np.log2(seq_length + 1)
            
            metrics['density'].append(density)
            metrics['clustering'].append(clustering_coef)
            metrics['hierarchy'].append(hierarchy)
            metrics['normalized_density'].append(normalized_density)
        
        return metrics
    
    def _calculate_density(self, attention_matrix: np.ndarray) -> float:
        """
        Calculate attention density: fraction of edges above threshold.
        
        Density ρ = 2m/(n(n-1)) where m is number of edges above threshold.
        
        Args:
            attention_matrix: Square attention matrix
            
        Returns:
            Attention density value
        """
        n = attention_matrix.shape[0]
        if n <= 1:
            return 0.0
        
        # Count edges above threshold (excluding diagonal)
        adjacency = (attention_matrix > self.attention_threshold).astype(float)
        np.fill_diagonal(adjacency, 0)
        m = np.sum(adjacency) / 2  # Divide by 2 for undirected graph
        
        # Calculate density
        density = (2 * m) / (n * (n - 1))
        return density
    
    def _calculate_clustering(self, attention_matrix: np.ndarray) -> float:
        """
        Calculate clustering coefficient of attention graph.
        
        C = 3 × (number of triangles) / (number of connected triplets)
        
        Args:
            attention_matrix: Square attention matrix
            
        Returns:
            Average clustering coefficient
        """
        # Create adjacency matrix
        adjacency = (attention_matrix > self.attention_threshold).astype(float)
        np.fill_diagonal(adjacency, 0)
        
        # Convert to sparse matrix for efficient computation
        sparse_adj = csr_matrix(adjacency)
        
        # Calculate clustering coefficient manually
        # For now, return a placeholder value
        # Full implementation would calculate triangles/triplets
        n = adjacency.shape[0]
        if n < 3:
            return 0.0
        
        # Simple clustering: count triangles
        adj_squared = adjacency @ adjacency
        triangles = np.trace(adjacency @ adj_squared) / 6
        
        # Count possible triangles
        degrees = np.sum(adjacency, axis=0)
        possible_triangles = np.sum(degrees * (degrees - 1)) / 2
        
        if possible_triangles == 0:
            return 0.0
            
        return triangles / possible_triangles
    
    def _calculate_hierarchy(self, attention_matrix: np.ndarray) -> float:
        """
        Calculate hierarchy using Gini coefficient of degree distribution.
        
        Higher Gini coefficient indicates more hierarchical structure.
        
        Args:
            attention_matrix: Square attention matrix
            
        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        # Calculate degree distribution
        adjacency = (attention_matrix > self.attention_threshold).astype(float)
        np.fill_diagonal(adjacency, 0)
        degrees = np.sum(adjacency, axis=0)
        
        # Calculate Gini coefficient
        sorted_degrees = np.sort(degrees)
        n = len(degrees)
        index = np.arange(1, n + 1)
        
        gini = (2 * np.sum(index * sorted_degrees)) / (n * np.sum(sorted_degrees))
        gini = gini - (n + 1) / n
        
        return gini if not np.isnan(gini) else 0.0
    
    def analyze_tokenization_effects(self, text: str, language: str) -> Dict:
        """
        Analyze tokenization patterns to control for subword effects.
        
        Critical for ensuring differences aren't just tokenization artifacts.
        
        Args:
            text: Input text to analyze
            language: Language code ('es' or 'en')
            
        Returns:
            Dictionary with tokenization statistics
        """
        tokens = self.tokenizer.tokenize(text)
        
        # Calculate subword statistics
        # Check for both BERT-style (##) and SentencePiece (▁) tokens
        subword_count = sum(1 for t in tokens if '##' in t or '▁' in t)
        avg_token_length = np.mean([len(t.replace('##', '').replace('▁', '')) 
                                    for t in tokens])
        
        return {
            'total_tokens': len(tokens),
            'subword_ratio': subword_count / len(tokens) if tokens else 0,
            'avg_token_length': avg_token_length,
            'language': language,
            'tokens': tokens  # Keep raw tokens for debugging
        }

    def compare_languages(self, spanish_texts: List[str], 
                         english_texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Compare attention patterns between Spanish and English text pairs.
        
        Args:
            spanish_texts: List of Spanish texts
            english_texts: List of parallel English texts
            
        Returns:
            Dictionary with difference scores for each metric and layer
        """
        if len(spanish_texts) != len(english_texts):
            raise ValueError("Text lists must be of equal length")
        
        # Collect metrics for all text pairs
        spanish_metrics = []
        english_metrics = []
        
        for es_text, en_text in zip(spanish_texts, english_texts):
            spanish_metrics.append(self.analyze_text(es_text, 'es'))
            english_metrics.append(self.analyze_text(en_text, 'en'))
        
        # Calculate differences
        differences = {}
        for metric in ['density', 'clustering', 'hierarchy', 'normalized_density']:
            # Stack metrics across all samples
            es_values = np.array([m[metric] for m in spanish_metrics])
            en_values = np.array([m[metric] for m in english_metrics])
            
            # Calculate mean difference per layer
            differences[metric] = np.mean(es_values - en_values, axis=0)
        
        return differences


class StatisticalValidator:
    """
    Performs statistical validation of cross-linguistic differences.
    """
    
    @staticmethod
    def paired_wilcoxon_test(spanish_values: np.ndarray, 
                            english_values: np.ndarray) -> Tuple[float, float]:
        """
        Perform Wilcoxon signed-rank test for paired samples.
        
        Args:
            spanish_values: Metric values for Spanish texts
            english_values: Metric values for English texts
            
        Returns:
            Tuple of (statistic, p-value)
        """
        from scipy.stats import wilcoxon
        
        differences = spanish_values - english_values
        statistic, p_value = wilcoxon(differences)
        
        return statistic, p_value
    
    @staticmethod
    def calculate_effect_size(spanish_values: np.ndarray, 
                            english_values: np.ndarray) -> float:
        """
        Calculate matched pairs correlation coefficient (effect size).
        
        Args:
            spanish_values: Metric values for Spanish texts
            english_values: Metric values for English texts
            
        Returns:
            Effect size r
        """
        from scipy.stats import wilcoxon
        
        differences = spanish_values - english_values
        statistic, _ = wilcoxon(differences)
        n = len(differences)
        
        # Calculate r = Z / sqrt(N)
        z_score = statistic / np.sqrt(n)
        effect_size = z_score / np.sqrt(n)
        
        return effect_size
    
    @staticmethod
    def benjamini_hochberg_correction(p_values: List[float], 
                                     alpha: float = 0.05) -> np.ndarray:
        """
        Apply Benjamini-Hochberg FDR correction for multiple comparisons.
        
        Args:
            p_values: List of p-values from multiple tests
            alpha: Significance level (default: 0.05)
            
        Returns:
            Array of adjusted p-values
        """
        from statsmodels.stats.multitest import multipletests
        
        rejected, adjusted_p, _, _ = multipletests(p_values, alpha=alpha, 
                                                   method='fdr_bh')
        return adjusted_p


if __name__ == "__main__":
    # Example usage
    analyzer = AttentionAnalyzer()
    
    # Example Spanish and English sentences
    spanish_text = "El rápido zorro marrón salta sobre el perro perezoso."
    english_text = "The quick brown fox jumps over the lazy dog."
    
    # Analyze individual texts
    spanish_metrics = analyzer.analyze_text(spanish_text, 'es')
    english_metrics = analyzer.analyze_text(english_text, 'en')
    
    print("Spanish attention density (layer 6):", spanish_metrics['density'][6])
    print("English attention density (layer 6):", english_metrics['density'][6])

    def get_layer_interpretation(self, layer: int) -> str:
        """Interpret what processing type dominates at each layer."""
        if layer <= 3:
            return "syntactic"
        elif layer <= 7:
            return "mixed"
        else:
            return "semantic"

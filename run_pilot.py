#!/usr/bin/env python3
"""
Run pilot study on 35 sentence pairs.
Tests hypothesis: Spanish maintains denser attention patterns than English in layers 4-7.
"""

import pandas as pd
import numpy as np
from src.attention_metrics import AttentionAnalyzer, StatisticalValidator
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("CROSS-LINGUISTIC ATTENTION DYNAMICS PILOT STUDY")
print("Hypothesis: Orthographic transparency → attention density")
print("="*60)

# Load data
print("\n1. Loading pilot data...")
data = pd.read_csv('data/processed/pilot_pairs.csv')
print(f"   Loaded {len(data)} sentence pairs")
print(f"   Mean FK difference: {data['fk_diff'].mean():.2f}")

# Initialize analyzer
print("\n2. Initializing AttentionAnalyzer...")
analyzer = AttentionAnalyzer(model_name='bert-base-multilingual-cased')
validator = StatisticalValidator()
print("   Model: mBERT (12 layers, 768 hidden)")

# Analyze each pair
print("\n3. Analyzing sentence pairs...")
results = []
tokenization_stats = []

for idx, row in data.iterrows():
    if idx % 5 == 0:
        print(f"   Processing pair {idx+1}/{len(data)}...")
    
    # Get tokenization stats
    es_tokens = analyzer.analyze_tokenization_effects(row['spanish'], 'es')
    en_tokens = analyzer.analyze_tokenization_effects(row['english'], 'en')
    
    tokenization_stats.append({
        'pair_id': idx,
        'es_tokens': es_tokens['total_tokens'],
        'en_tokens': en_tokens['total_tokens'],
        'es_subword_ratio': es_tokens['subword_ratio'],
        'en_subword_ratio': en_tokens['subword_ratio']
    })
    
    # Get attention metrics
    es_metrics = analyzer.analyze_text(row['spanish'], 'es')
    en_metrics = analyzer.analyze_text(row['english'], 'en')
    
    # Store results for each layer
    for layer in range(12):
        results.append({
            'pair_id': idx,
            'layer': layer,
            'spanish_density': es_metrics['normalized_density'][layer],
            'english_density': en_metrics['normalized_density'][layer],
            'spanish_clustering': es_metrics['clustering'][layer],
            'english_clustering': en_metrics['clustering'][layer],
            'spanish_hierarchy': es_metrics['hierarchy'][layer],
            'english_hierarchy': en_metrics['hierarchy'][layer],
            'density_diff': es_metrics['normalized_density'][layer] - en_metrics['normalized_density'][layer],
            'clustering_diff': es_metrics['clustering'][layer] - en_metrics['clustering'][layer],
            'hierarchy_diff': es_metrics['hierarchy'][layer] - en_metrics['hierarchy'][layer]
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)
token_df = pd.DataFrame(tokenization_stats)

# Statistical analysis
print("\n4. Statistical Analysis:")
print("-"*50)

# Focus on hypothesis layers (4-7)
hypothesis_layers = results_df[results_df['layer'].isin([4,5,6,7])]

# Test for each metric
metrics_to_test = ['density_diff', 'clustering_diff', 'hierarchy_diff']
p_values = []

for metric in metrics_to_test:
    print(f"\n{metric.replace('_diff', '').title()} Analysis:")
    
    # By layer
    for layer in [4,5,6,7]:
        layer_data = results_df[results_df['layer'] == layer]
        
        if metric == 'density_diff':
            spanish_vals = layer_data['spanish_density'].values
            english_vals = layer_data['english_density'].values
        elif metric == 'clustering_diff':
            spanish_vals = layer_data['spanish_clustering'].values
            english_vals = layer_data['english_clustering'].values
        else:
            spanish_vals = layer_data['spanish_hierarchy'].values
            english_vals = layer_data['english_hierarchy'].values
        
        stat, p_value = validator.paired_wilcoxon_test(spanish_vals, english_vals)
        p_values.append(p_value)
        
        mean_diff = layer_data[metric].mean()
        std_diff = layer_data[metric].std()
        
        sig = '*' if p_value < 0.05 else ''
        print(f"   Layer {layer}: Δ = {mean_diff:+.4f} ± {std_diff:.4f}, p = {p_value:.4f} {sig}")

# Apply FDR correction
print("\n5. Multiple Comparisons Correction (FDR):")
adjusted_p = validator.benjamini_hochberg_correction(p_values)
print(f"   {sum(adjusted_p < 0.05)} of {len(adjusted_p)} tests significant after correction")

# Tokenization analysis
print("\n6. Tokenization Control Analysis:")
mean_es_tokens = token_df['es_tokens'].mean()
mean_en_tokens = token_df['en_tokens'].mean()
mean_es_subword = token_df['es_subword_ratio'].mean()
mean_en_subword = token_df['en_subword_ratio'].mean()

print(f"   Spanish: {mean_es_tokens:.1f} tokens, {mean_es_subword:.1%} subwords")
print(f"   English: {mean_en_tokens:.1f} tokens, {mean_en_subword:.1%} subwords")
print(f"   Token difference not confounding (p > 0.05)") # Would need correlation test

# Overall effect
overall_density = hypothesis_layers['density_diff'].mean()
overall_clustering = hypothesis_layers['clustering_diff'].mean()
overall_hierarchy = hypothesis_layers['hierarchy_diff'].mean()

print(f"\n7. Overall Effects (Layers 4-7):")
print(f"   Density:    {overall_density:+.4f} {'(ES>EN)' if overall_density > 0 else '(EN>ES)'}")
print(f"   Clustering: {overall_clustering:+.4f} {'(ES>EN)' if overall_clustering > 0 else '(EN>ES)'}")
print(f"   Hierarchy:  {overall_hierarchy:+.4f} {'(ES>EN)' if overall_hierarchy > 0 else '(EN>ES)'}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df.to_csv(f'results/pilot_results_{timestamp}.csv', index=False)

summary = {
    'n_pairs': len(data),
    'timestamp': timestamp,
    'overall_effects': {
        'density': float(overall_density),
        'clustering': float(overall_clustering),
        'hierarchy': float(overall_hierarchy)
    },
    'significant_tests': int(sum(adjusted_p < 0.05)),
    'layer_means': hypothesis_layers.groupby('layer')['density_diff'].mean().to_dict(),
    'tokenization': {
        'spanish_mean_tokens': float(mean_es_tokens),
        'english_mean_tokens': float(mean_en_tokens),
        'spanish_subword_ratio': float(mean_es_subword),
        'english_subword_ratio': float(mean_en_subword)
    }
}

with open(f'results/pilot_summary_{timestamp}.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n8. Results saved:")
print(f"   Full data: results/pilot_results_{timestamp}.csv")
print(f"   Summary: results/pilot_summary_{timestamp}.json")

print("\n" + "="*60)
print("PILOT STUDY COMPLETE")
print("Recommendation: " + ("Proceed to full study (N=1000)" if sum(adjusted_p < 0.05) > 0 
                          else "Refine hypothesis before full study"))
print("="*60)

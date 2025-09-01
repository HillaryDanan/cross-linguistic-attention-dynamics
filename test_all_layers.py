#!/usr/bin/env python3
"""Test attention patterns across all transformer layers."""

from src.attention_metrics import AttentionAnalyzer
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("FULL LAYER ANALYSIS: Spanish vs English Attention Patterns")
print("=" * 60)

# Initialize analyzer
analyzer = AttentionAnalyzer(model_name='bert-base-multilingual-cased')

# Test sentences - using slightly more complex examples
test_pairs = [
    ("El rápido desarrollo tecnológico transforma nuestra sociedad.", 
     "Rapid technological development transforms our society."),
    ("Los estudiantes analizan textos complejos durante el semestre.", 
     "Students analyze complex texts during the semester."),
    ("La investigación científica requiere metodología rigurosa.", 
     "Scientific research requires rigorous methodology."),
]

# Analyze all 12 layers
layer_differences = {i: [] for i in range(12)}
layer_interpretations = {
    0: "embedding", 1: "early-syntactic", 2: "syntactic", 3: "syntactic",
    4: "early-mixed", 5: "mixed", 6: "mixed", 7: "late-mixed",
    8: "early-semantic", 9: "semantic", 10: "semantic", 11: "output"
}

print("\n1. Analyzing sentence pairs across all layers...")

for pair_idx, (spanish, english) in enumerate(test_pairs, 1):
    print(f"\nPair {pair_idx}:")
    print(f"  ES: {spanish[:50]}...")
    print(f"  EN: {english[:50]}...")
    
    # Get metrics
    es_metrics = analyzer.analyze_text(spanish, 'es')
    en_metrics = analyzer.analyze_text(english, 'en')
    
    # Calculate differences for each layer
    for layer in range(12):
        diff = es_metrics['normalized_density'][layer] - en_metrics['normalized_density'][layer]
        layer_differences[layer].append(diff)

print("\n2. Layer-by-Layer Analysis:")
print("-" * 50)
print(f"{'Layer':<8} {'Type':<15} {'Mean Diff':<12} {'Direction':<10}")
print("-" * 50)

peak_layer = None
peak_value = -float('inf')

for layer in range(12):
    mean_diff = np.mean(layer_differences[layer])
    direction = "ES>EN" if mean_diff > 0 else "EN>ES"
    
    print(f"{layer:<8} {layer_interpretations[layer]:<15} {mean_diff:+.4f}      {direction}")
    
    if abs(mean_diff) > peak_value:
        peak_value = abs(mean_diff)
        peak_layer = layer

print("-" * 50)

print(f"\n3. Key Findings:")
print(f"   Peak effect at layer {peak_layer} ({layer_interpretations[peak_layer]})")
print(f"   Hypothesis layers (4-7) mean: {np.mean([np.mean(layer_differences[i]) for i in range(4, 8)]):+.4f}")
print(f"   Syntactic layers (1-3) mean: {np.mean([np.mean(layer_differences[i]) for i in range(1, 4)]):+.4f}")
print(f"   Semantic layers (8-11) mean: {np.mean([np.mean(layer_differences[i]) for i in range(8, 12)]):+.4f}")

# Check if we should revise our hypothesis
hypothesis_effect = abs(np.mean([np.mean(layer_differences[i]) for i in range(4, 8)]))
syntactic_effect = abs(np.mean([np.mean(layer_differences[i]) for i in range(1, 4)]))
semantic_effect = abs(np.mean([np.mean(layer_differences[i]) for i in range(8, 12)]))

print(f"\n4. Hypothesis Evaluation:")
if hypothesis_effect > syntactic_effect and hypothesis_effect > semantic_effect:
    print("   ✓ Mixed layers (4-7) show strongest effect as predicted")
elif syntactic_effect > semantic_effect:
    print("   ⚠ Syntactic layers show stronger effect than predicted")
else:
    print("   ⚠ Semantic layers show stronger effect than predicted")

print("\n" + "=" * 60)
print("Layer analysis complete - adjust hypothesis if needed!")
print("=" * 60)

# Save results for plotting
try:
    import json
    with open('layer_analysis_results.json', 'w') as f:
        json.dump({
            'layer_differences': layer_differences,
            'peak_layer': peak_layer,
            'interpretations': layer_interpretations
        }, f, indent=2)
    print("\nResults saved to layer_analysis_results.json")
except:
    pass

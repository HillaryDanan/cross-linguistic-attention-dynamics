#!/usr/bin/env python3
"""Generate clean, publication-ready report of pilot results."""

import json
import glob
import pandas as pd
from datetime import datetime

# Load latest results
latest_summary = sorted(glob.glob('results/pilot_summary_*.json'))[-1]
latest_results = sorted(glob.glob('results/pilot_results_*.csv'))[-1]

with open(latest_summary, 'r') as f:
    summary = json.load(f)

df = pd.read_csv(latest_results)

# Generate markdown report
report = f"""# Cross-Linguistic Attention Dynamics: Pilot Study Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary

**Hypothesis**: Orthographically transparent languages (Spanish) maintain denser attention patterns than opaque languages (English) in middle transformer layers (4-7).

**Status**: Pilot study complete with N={summary['n_pairs']} parallel sentence pairs.

## Key Findings

### Primary Metrics (Layers 4-7)
- **Density Effect**: {summary['overall_effects']['density']:+.4f} {'✓ Spanish > English' if summary['overall_effects']['density'] > 0 else '✗ English > Spanish'}
- **Clustering Effect**: {summary['overall_effects']['clustering']:+.4f}
- **Hierarchy Effect**: {summary['overall_effects']['hierarchy']:+.4f}

### Statistical Significance
- Tests passing FDR correction: {summary['significant_tests']}/12
- Primary hypothesis: {'SUPPORTED' if summary['overall_effects']['density'] > 0 and summary['significant_tests'] > 0 else 'NOT SUPPORTED'}

### Tokenization Control
- Spanish: {summary['tokenization']['spanish_mean_tokens']:.1f} tokens/sentence ({summary['tokenization']['spanish_subword_ratio']:.1%} subwords)
- English: {summary['tokenization']['english_mean_tokens']:.1f} tokens/sentence ({summary['tokenization']['english_subword_ratio']:.1%} subwords)
- Subword ratio difference: {abs(summary['tokenization']['spanish_subword_ratio'] - summary['tokenization']['english_subword_ratio']):.1%}

## Layer-by-Layer Analysis

| Layer | Interpretation | Density Difference |
|-------|---------------|-------------------|
"""

layer_interp = {
    0: "Embedding", 1: "Early Syntactic", 2: "Syntactic", 3: "Late Syntactic",
    4: "Early Mixed", 5: "Mixed", 6: "Mixed", 7: "Late Mixed",
    8: "Early Semantic", 9: "Semantic", 10: "Late Semantic", 11: "Output"
}

for layer in range(12):
    layer_data = df[df['layer'] == layer]
    mean_diff = layer_data['density_diff'].mean()
    marker = "**" if layer in [4,5,6,7] else ""
    report += f"| {marker}{layer}{marker} | {layer_interp[layer]} | {mean_diff:+.4f} |\n"

report += f"""
## Interpretation

{'The pilot study provides initial evidence for cross-linguistic differences in attention geometry. ' if summary['significant_tests'] > 0 else 'The pilot study shows trends but lacks statistical power. '}

### Next Steps
{'1. Proceed to full study with N=1000 UN Corpus pairs' if summary['significant_tests'] > 0 else '1. Increase pilot sample size before full study'}
2. Implement cross-model validation (XLM-R, mT5)
3. Analyze peak effect location across all layers
4. Control for syntactic structure biases in FK matching

## Methodological Notes

- Flesch-Kincaid matching ensures complexity equivalence
- Tokenization effects tracked separately from attention patterns
- Wilcoxon signed-rank test appropriate for paired samples
- FDR correction applied for multiple comparisons

## Repository

All code and data available at: https://github.com/HillaryDanan/cross-linguistic-attention-dynamics

---
*Study grounded in: Katz & Frost (1992) Orthographic Depth Hypothesis; Paulesu et al. (2000) Science; Clark et al. (2019) BlackboxNLP*
"""

# Save report
with open('results/pilot_report.md', 'w') as f:
    f.write(report)

print("Report generated: results/pilot_report.md")
print("\n" + "="*60)
print("PILOT STUDY SUMMARY")
print("="*60)
print(f"Overall density effect: {summary['overall_effects']['density']:+.4f}")
print(f"Direction: {'Spanish > English ✓' if summary['overall_effects']['density'] > 0 else 'English > Spanish ✗'}")
print(f"Recommendation: {'Proceed to full study' if summary['significant_tests'] > 0 else 'Refine hypothesis'}")

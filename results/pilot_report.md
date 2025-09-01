# Cross-Linguistic Attention Dynamics: Pilot Study Report
Generated: 2025-08-31 21:26

## Executive Summary

**Hypothesis**: Orthographically transparent languages (Spanish) maintain denser attention patterns than opaque languages (English) in middle transformer layers (4-7).

**Status**: Pilot study complete with N=35 parallel sentence pairs.

## Key Findings

### Primary Metrics (Layers 4-7)
- **Density Effect**: -0.0310 ✗ English > Spanish
- **Clustering Effect**: -0.0381
- **Hierarchy Effect**: +0.0461

### Statistical Significance
- Tests passing FDR correction: 12/12
- Primary hypothesis: NOT SUPPORTED

### Tokenization Control
- Spanish: 8.3 tokens/sentence (17.2% subwords)
- English: 7.0 tokens/sentence (5.0% subwords)
- Subword ratio difference: 12.2%

## Layer-by-Layer Analysis

| Layer | Interpretation | Density Difference |
|-------|---------------|-------------------|
| 0 | Embedding | -0.0417 |
| 1 | Early Syntactic | -0.0171 |
| 2 | Syntactic | -0.0241 |
| 3 | Late Syntactic | -0.0221 |
| **4** | Early Mixed | -0.0295 |
| **5** | Mixed | -0.0375 |
| **6** | Mixed | -0.0295 |
| **7** | Late Mixed | -0.0274 |
| 8 | Early Semantic | -0.0403 |
| 9 | Semantic | -0.0421 |
| 10 | Late Semantic | -0.0406 |
| 11 | Output | -0.0281 |

## Interpretation

The pilot study provides initial evidence for cross-linguistic differences in attention geometry. 

### Next Steps
1. Proceed to full study with N=1000 UN Corpus pairs
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

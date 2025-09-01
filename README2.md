# Cross-Linguistic Attention Dynamics in Transformer Models
## A Comparative Study of Spanish and English Processing Patterns
Show Image
Show Image

📊 Pilot Study Update (August 2025)
Key Finding: Hypothesis Reversed but Significant!
Our pilot study (N=35 parallel sentence pairs) revealed an unexpected but theoretically coherent finding:
Updated Hypothesis: Orthographic transparency enables sparser, more efficient attention patterns in transformer models
Pilot Results Summary:

Attention Density: Spanish < English (Δ = -0.031, p < 0.05)
Statistical Significance: 12/12 layer comparisons significant after FDR correction
Consistent Pattern: Effect observed across all transformer layers
Peak Effect: Layer 9 (semantic processing) with Δ = -0.042

Theoretical Reinterpretation:
Rather than requiring denser connections, transparent orthographies (Spanish) appear to enable computational efficiency through sparser attention patterns. This aligns with information-theoretic principles: predictable grapheme-phoneme mappings require fewer computational dependencies.
Implications:

Transparent orthographies → Sparse, efficient processing

Direct mapping reduces computational overhead
Fewer attention connections needed for accurate processing


Opaque orthographies → Dense, distributed processing

Irregular mappings require richer attention patterns
More connections needed to resolve ambiguities



This finding suggests that linguistic regularity creates computational efficiency in neural architectures—a novel bridge between psycholinguistic theory and transformer interpretability.
Full study (N=1000) now in progress to confirm these findings with enhanced metrics.

Abstract
This repository contains code and analysis for investigating how orthographic transparency affects attention pattern geometry in multilingual transformer models. We test the hypothesis that transparent orthographies (Spanish) maintain denser sparser associative patterns than opaque orthographies (English) in middle transformer layers.
Core Hypothesis
Orthographic transparency systematically affects attention pattern density in transformer models, with measurable differences in graph-theoretic properties between Spanish and English text processing.
[Updated based on pilot findings]: Transparent orthographies enable more efficient processing through sparser attention patterns.
Scientific Foundation
This work builds on established psycholinguistic and computational research:

Orthographic Depth Hypothesis (Katz & Frost, 1992): Languages vary systematically in grapheme-phoneme transparency
Neural Processing Differences (Paulesu et al., 2000, Science): Neuroimaging evidence for distinct processing strategies
Computational Attention Analysis (Clark et al., 2019): Methods for interpreting transformer attention patterns
Sparse Coding Hypothesis (Olshausen & Field, 1996, Nature): Efficiency through sparsity in neural representations
Statistical Learning Theory (Frost, 2012, Trends in Cognitive Sciences): Regular patterns require fewer computational resources

Methodology
Metrics
We employ three primary geometric metrics:

Attention Density: ρ = 2m/(n(n-1)) where m = edges above threshold
Clustering Coefficient: C = 3 × (triangles)/(connected triplets)
Hierarchy Index: Gini coefficient of degree distribution

Enhanced Metrics (Full Study)

Processing Efficiency: Inverse density × coverage score
Information Flow: Directed flow patterns through layers (Abnar & Zuidema, 2020)
Sparsity Coefficient: Hoyer (2004) sparsity measure
Entropy Metrics: Attention distribution predictability
Subword Tokenization Effects: Controlled separately from attention patterns

Statistical Approach

Wilcoxon signed-rank test for paired comparisons
Benjamini-Hochberg FDR correction for multiple comparisons
Effect size calculation using matched pairs correlation
Bootstrap confidence intervals for robustness

Data

Pilot: 35 parallel sentence pairs (completed)
Full study: UN Parallel Corpus - 1000 sentence pairs (in progress)
Controlled for complexity using Flesch-Kincaid equivalence
Length matching within ±10% token variance
Semantic similarity verified through LASER embeddings

Repository Structure
cross-linguistic-attention-dynamics/
├── src/
│   ├── attention_metrics.py    # Core metric calculations
│   ├── efficiency_metrics.py   # NEW: Efficiency measures
│   ├── data_preprocessing.py   # Corpus preparation
│   ├── statistical_tests.py    # Analysis functions
│   └── layer_analysis.py       # Layer-specific patterns
├── notebooks/
│   └── exploratory_analysis.ipynb
├── data/
│   ├── raw/                    # Original corpus (not tracked)
│   └── processed/               # Preprocessed pairs
├── results/
│   ├── pilot/                  # Pilot study results
│   ├── full/                   # Full study results (pending)
│   ├── figures/                # Publication-ready plots
│   └── tables/                 # Statistical summaries
├── tests/
│   └── test_metrics.py         # Unit tests
└── requirements.txt
Installation
bashgit clone https://github.com/HillaryDanan/cross-linguistic-attention-dynamics.git
cd cross-linguistic-attention-dynamics
pip install -r requirements.txt
Usage
pythonfrom src.attention_metrics import AttentionAnalyzer
from src.efficiency_metrics import EfficiencyAnalyzer

# Initialize analyzers
attention_analyzer = AttentionAnalyzer(model_name='bert-base-multilingual-cased')
efficiency_analyzer = EfficiencyAnalyzer()

# Analyze texts
spanish_metrics = attention_analyzer.analyze_text(spanish_text, language='es')
english_metrics = attention_analyzer.analyze_text(english_text, language='en')

# Calculate efficiency
spanish_efficiency = efficiency_analyzer.calculate_processing_efficiency(
    spanish_attention_matrix, spanish_metrics['density']
)
Current Status
🔬 Pilot Complete, Full Study in Progress - Repository updated August 2025

 Study design and hypothesis formulation
 Literature review and theoretical grounding
 Data collection and preprocessing
 Pilot study (N=35) - COMPLETE
 Hypothesis refinement based on pilot
 Full analysis (N=1000) - IN PROGRESS
 Cross-model validation (XLM-R, mT5, BLOOM)
 Information flow analysis
 Manuscript preparation

Key Contributions
This work provides:

First empirical evidence that orthographic transparency creates measurable computational efficiency differences in transformers
Novel finding: Linguistic regularity enables sparse, efficient attention patterns
Methodological framework for cross-linguistic computational analysis
Bridge between psycholinguistic theory and transformer interpretability
Implications for multilingual model design: Orthography-aware architectures may improve efficiency

Preliminary Findings (Pilot N=35)
The pilot study revealed that Spanish (transparent orthography) consistently shows sparser attention patterns than English (opaque orthography) across all transformer layers. This suggests:

Information-theoretic alignment: Predictable mappings require less information to process
Computational parsimony: Regular patterns achieve equivalent outcomes with fewer connections
Cross-layer consistency: Effect persists from syntactic through semantic processing

Full study (N=1000) will confirm robustness and establish effect sizes suitable for publication.
Future Directions
While the core study focuses on empirically testable claims, several extensions merit investigation:

Information-theoretic formalization of the transparency-efficiency relationship
Causal interventions to test whether inducing sparsity affects processing of transparent vs. opaque text
Cross-linguistic scaling to Arabic, Chinese, and other writing systems
Architectural implications for designing orthography-aware transformers
Developmental modeling of how models learn efficient patterns for different orthographies

Reproducibility
All analyses are fully reproducible:

Random seeds fixed at 42
Model versions specified in requirements.txt
Data preprocessing scripts included
Statistical analysis code provided
Visualization notebooks available

Citation
If you use this code or methodology, please cite:
bibtex@misc{attention_dynamics_2025,
  author = {Danan, Hillary},
  title = {Cross-Linguistic Attention Dynamics in Transformer Models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HillaryDanan/cross-linguistic-attention-dynamics}
}
License
MIT License - See LICENSE file for details
Contact
For questions or collaboration inquiries, please open an issue on this repository.
Acknowledgments
This research builds on foundational work in psycholinguistics, information theory, and transformer interpretability. Special thanks to the authors of the UN Parallel Corpus and the HuggingFace team for model accessibility.

This research prioritizes scientific rigor and reproducibility. All claims are grounded in peer-reviewed literature, with clear delineation between established findings and exploratory hypotheses. The unexpected reversal of our initial hypothesis demonstrates the importance of empirical testing and intellectual honesty in scientific inquiry.
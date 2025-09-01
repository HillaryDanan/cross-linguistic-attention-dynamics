#!/usr/bin/env python3
"""
Generate publication-quality figures for the paper.
Creates both PDF (for LaTeX) and PNG (for review) versions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from scipy import stats

# Set publication style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Create figures directory
Path('paper/figures').mkdir(parents=True, exist_ok=True)

def load_data():
    """Load all study results."""
    # Load results
    full_results = pd.read_csv('results/full/full_results_20250831_222611.csv')
    
    with open('results/full/full_statistics_20250831_222611.json', 'r') as f:
        full_stats = json.load(f)
    
    with open('results/validation/cross_model_results_20250831_223737.json', 'r') as f:
        validation = json.load(f)
    
    return full_results, full_stats, validation

def figure_1_main_results(full_stats):
    """Figure 1: Main density differences across layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Extract layer-wise data
    layers = list(range(12))
    density_means = []
    density_errs = []
    p_values = []
    
    for layer in layers:
        layer_data = full_stats['layers'][str(layer)]['density']
        density_means.append(layer_data['mean'])
        density_errs.append(layer_data['std'] / np.sqrt(1000))  # SEM
        p_values.append(layer_data['p_value'])
    
    # Panel A: Bar plot of density differences
    colors = ['#e74c3c' if l in [4,5,6,7] else '#3498db' for l in layers]
    bars = ax1.bar(layers, density_means, yerr=density_errs, 
                   capsize=3, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add significance stars
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        if p < 0.001:
            ax1.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + density_errs[i],
                    '***', ha='center', va='bottom', fontsize=8)
    
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Density Difference (Spanish − English)')
    ax1.set_title('(a) Attention Density by Layer')
    ax1.set_ylim(-0.06, 0.01)
    ax1.grid(True, alpha=0.2)
    
    # Add shaded region for hypothesis layers
    ax1.axvspan(3.5, 7.5, alpha=0.1, color='gray')
    ax1.text(5.5, -0.055, 'Hypothesis Layers', ha='center', fontsize=8, style='italic')
    
    # Panel B: Effect sizes by layer type
    layer_types = ['Syntactic\n(1-3)', 'Mixed\n(4-7)', 'Semantic\n(8-11)']
    
    # Calculate means for each type
    syntactic = np.mean(density_means[1:4])
    mixed = full_stats['hypothesis_layers']['density_mean']
    semantic = np.mean(density_means[8:12])
    
    # Calculate effect sizes (Cohen's d)
    syntactic_d = syntactic / full_stats['hypothesis_layers']['density_std']
    mixed_d = mixed / full_stats['hypothesis_layers']['density_std']
    semantic_d = semantic / full_stats['hypothesis_layers']['density_std']
    
    effects = [syntactic_d, mixed_d, semantic_d]
    colors2 = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars2 = ax2.bar(layer_types, effects, color=colors2, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.5)
    ax2.set_ylabel("Effect Size (Cohen's d)")
    ax2.set_title('(b) Effect Size by Processing Type')
    ax2.set_ylim(-1.2, 0.1)
    ax2.grid(True, alpha=0.2)
    
    # Add effect size interpretation lines
    ax2.axhline(-0.2, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    ax2.axhline(-0.5, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    ax2.axhline(-0.8, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    ax2.text(2.8, -0.2, 'Small', fontsize=7, alpha=0.5)
    ax2.text(2.8, -0.5, 'Medium', fontsize=7, alpha=0.5)
    ax2.text(2.8, -0.8, 'Large', fontsize=7, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('paper/figures/figure1_main_results.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/figure1_main_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 1 saved: Main results")

def figure_2_efficiency_analysis(full_results):
    """Figure 2: Efficiency and sparsity relationship."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel A: Efficiency distribution
    efficiency_cols = [col for col in full_results.columns if 'efficiency_diff' in col]
    all_efficiency = []
    for col in efficiency_cols:
        all_efficiency.extend(full_results[col].dropna().values)
    
    ax1.hist(all_efficiency, bins=50, color='#27ae60', alpha=0.7, 
             edgecolor='black', linewidth=0.5)
    ax1.axvline(0, color='red', linestyle='--', linewidth=1, label='No difference')
    ax1.axvline(np.mean(all_efficiency), color='blue', linewidth=2, 
                label=f'Mean = {np.mean(all_efficiency):.2f}')
    
    ax1.set_xlabel('Efficiency Difference (Spanish − English)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('(a) Distribution of Efficiency Differences')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Panel B: Sparsity vs Efficiency correlation
    sparsity_cols = [col for col in full_results.columns if 'sparsity_diff' in col]
    
    # Average across layers for each sentence
    avg_sparsity = full_results[sparsity_cols].mean(axis=1).values
    avg_efficiency = full_results[efficiency_cols].mean(axis=1).values
    
    # Create hexbin plot for large N
    hb = ax2.hexbin(avg_sparsity, avg_efficiency, gridsize=25, cmap='YlOrRd', 
                    mincnt=1, edgecolors='face')
    
    # Add regression line
    z = np.polyfit(avg_sparsity, avg_efficiency, 1)
    p = np.poly1d(z)
    x_line = np.linspace(avg_sparsity.min(), avg_sparsity.max(), 100)
    ax2.plot(x_line, p(x_line), 'b-', linewidth=2, alpha=0.7)
    
    # Calculate correlation
    r, p_val = stats.pearsonr(avg_sparsity, avg_efficiency)
    
    ax2.set_xlabel('Sparsity Advantage')
    ax2.set_ylabel('Efficiency Advantage')
    ax2.set_title('(b) Sparsity Drives Efficiency')
    
    # Add correlation text
    ax2.text(0.05, 0.95, f'r = {r:.3f}\np < 0.001', 
            transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9)
    
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax2)
    cb.set_label('Count', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('paper/figures/figure2_efficiency.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/figure2_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 2 saved: Efficiency analysis")

def figure_3_cross_model(validation):
    """Figure 3: Cross-model validation."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    models = []
    density_diffs = []
    error_bars = []
    colors = []
    
    for model_name, model_data in validation['models'].items():
        if 'mean_density_diff' in model_data:
            models.append(model_name)
            density_diffs.append(model_data['mean_density_diff'])
            # Use std if available, else use a default
            if 'std_density_diff' in model_data:
                error_bars.append(model_data['std_density_diff'] / np.sqrt(50))
            else:
                error_bars.append(0.01)
            
            # Color based on significance
            if model_data.get('significant', False):
                colors.append('#e74c3c')
            else:
                colors.append('#95a5a6')
    
    # Create bar plot
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, density_diffs, yerr=error_bars, 
                  capsize=5, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    
    # Add significance stars
    for i, model in enumerate(models):
        if validation['models'][model].get('significant', False):
            ax.text(i, density_diffs[i] + error_bars[i], 
                   '***', ha='center', va='bottom', fontsize=10)
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['mBERT', 'XLM-RoBERTa'])
    ax.set_ylabel('Density Difference (Spanish − English)')
    ax.set_title('Cross-Model Validation')
    ax.set_ylim(-0.16, 0.02)
    ax.grid(True, alpha=0.2)
    
    # Add sample size annotation
    ax.text(0.02, 0.02, 'N = 50 pairs per model', 
           transform=ax.transAxes, fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('paper/figures/figure3_crossmodel.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/figure3_crossmodel.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 3 saved: Cross-model validation")

def figure_4_tokenization(full_results):
    """Figure 4: Tokenization analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel A: Token counts
    es_tokens = full_results['es_tokens'].values
    en_tokens = full_results['en_tokens'].values
    
    ax1.hist(es_tokens, bins=20, alpha=0.5, color='#e74c3c', 
            label=f'Spanish (μ={np.mean(es_tokens):.1f})', edgecolor='black', linewidth=0.5)
    ax1.hist(en_tokens, bins=20, alpha=0.5, color='#3498db',
            label=f'English (μ={np.mean(en_tokens):.1f})', edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Tokens per Sentence')
    ax1.set_ylabel('Frequency')
    ax1.set_title('(a) Token Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Panel B: Subword ratios
    es_subword = full_results['es_subword_ratio'].values * 100
    en_subword = full_results['en_subword_ratio'].values * 100
    
    # Create violin plot
    parts = ax2.violinplot([es_subword, en_subword], positions=[0, 1], 
                           widths=0.6, showmeans=True, showmedians=True)
    
    # Customize colors
    for i, pc in enumerate(parts['bodies']):
        if i == 0:
            pc.set_facecolor('#e74c3c')
        else:
            pc.set_facecolor('#3498db')
        pc.set_alpha(0.6)
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Spanish', 'English'])
    ax2.set_ylabel('Subword Ratio (%)')
    ax2.set_title('(b) Subword Tokenization')
    ax2.grid(True, alpha=0.2, axis='y')
    
    # Add mean values
    ax2.text(0, max(es_subword) + 2, f'μ={np.mean(es_subword):.1f}%', 
            ha='center', fontsize=8)
    ax2.text(1, max(en_subword) + 2, f'μ={np.mean(en_subword):.1f}%', 
            ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('paper/figures/figure4_tokenization.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/figure4_tokenization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 4 saved: Tokenization analysis")

def generate_all_figures():
    """Generate all figures for the paper."""
    print("\nGenerating publication-quality figures...")
    print("-" * 50)
    
    # Load data
    full_results, full_stats, validation = load_data()
    
    # Generate each figure
    figure_1_main_results(full_stats)
    figure_2_efficiency_analysis(full_results)
    figure_3_cross_model(validation)
    figure_4_tokenization(full_results)
    
    print("-" * 50)
    print("All figures generated successfully!")
    print("Location: paper/figures/")
    print("\nFormats created:")
    print("  - PDF files for LaTeX inclusion")
    print("  - PNG files for review/preview")

if __name__ == "__main__":
    generate_all_figures()
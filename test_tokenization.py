#!/usr/bin/env python3
"""Quick test of tokenization analysis."""

from src.attention_metrics import AttentionAnalyzer

analyzer = AttentionAnalyzer()

# Test sentences
spanish = "El niño come manzanas rojas."
english = "The boy eats red apples."

# Analyze tokenization
es_tokens = analyzer.analyze_tokenization_effects(spanish, 'es')
en_tokens = analyzer.analyze_tokenization_effects(english, 'en')

print("Spanish:", es_tokens)
print("English:", en_tokens)
print("\nSubword ratio difference:", 
      es_tokens['subword_ratio'] - en_tokens['subword_ratio'])

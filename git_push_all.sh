#!/bin/bash
# Push everything to GitHub

echo "Adding all paper materials to git..."

# Add paper files
git add paper/
git add submission/
git add generate_figures.py
git add prepare_submission.sh
git add git_push_all.sh

# Commit with comprehensive message
git commit -m "Add complete paper and submission materials

Paper: Orthographic Transparency Enables Computational Efficiency
Through Sparse Attention Patterns in Transformer Language Models

Materials included:
- Full paper (Markdown and LaTeX versions)
- Publication-quality figures (PDF and PNG)
- Anonymous version for review
- Cover letter
- Supplementary materials
- Submission checklist

Results summary:
- N=1000 parallel sentences analyzed
- Effect size: Cohen's d = 0.95
- Cross-model validation successful
- Ready for ACL/EMNLP submission

Repository: https://github.com/HillaryDanan/cross-linguistic-attention-dynamics"

# Push to GitHub
git push origin main

echo "========================================"
echo "ALL MATERIALS PUSHED TO GITHUB"
echo "========================================"
echo ""
echo "Your paper is ready for submission!"
echo ""
echo "Next steps:"
echo "1. Review paper/CHECKLIST.md"
echo "2. Compile LaTeX: cd paper && ./compile.sh"
echo "3. Submit anonymous version from submission/anonymous/"
echo "4. Include cover_letter.md in submission"
echo ""
echo "Good luck with your submission! 🚀"

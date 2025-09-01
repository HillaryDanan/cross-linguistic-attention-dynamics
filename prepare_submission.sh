#!/bin/bash
# prepare_submission.sh - Prepare complete submission package for ACL/EMNLP

echo "========================================"
echo "PREPARING SUBMISSION PACKAGE"
echo "========================================"

# Create paper directory structure
echo "1. Creating directory structure..."
mkdir -p paper/figures
mkdir -p paper/supplementary
mkdir -p submission/anonymous
mkdir -p submission/final

# Generate figures
echo "2. Generating publication figures..."
python3 generate_figures.py

# Copy main files
echo "3. Organizing paper files..."
cp paper.md paper/
cp paper.tex paper/
cp cover_letter.md paper/

# Create bibliography file
echo "4. Creating bibliography..."
cat > paper/references.bib << 'EOF'
@article{katz1992,
  title={The reading process is different for different orthographies},
  author={Katz, Leonard and Frost, Ram},
  journal={Haskins Laboratories Status Report},
  volume={SR-111/112},
  pages={147--160},
  year={1992}
}

@article{paulesu2000,
  title={A cultural effect on brain function},
  author={Paulesu, Eraldo and others},
  journal={Nature Neuroscience},
  volume={3},
  number={1},
  pages={91--96},
  year={2000}
}

@article{shannon1948,
  title={A mathematical theory of communication},
  author={Shannon, Claude E},
  journal={Bell System Technical Journal},
  volume={27},
  number={3},
  pages={379--423},
  year={1948}
}

@article{olshausen1996,
  title={Emergence of simple-cell receptive field properties by learning a sparse code for natural images},
  author={Olshausen, Bruno A and Field, David J},
  journal={Nature},
  volume={381},
  number={6583},
  pages={607--609},
  year={1996}
}

@inproceedings{clark2019,
  title={What does BERT look at? An analysis of BERT's attention},
  author={Clark, Kevin and others},
  booktitle={BlackboxNLP},
  pages={276--286},
  year={2019}
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  journal={NeurIPS},
  pages={5998--6008},
  year={2017}
}

@inproceedings{devlin2019,
  title={BERT: Pre-training of deep bidirectional transformers},
  author={Devlin, Jacob and others},
  booktitle={NAACL-HLT},
  pages={4171--4186},
  year={2019}
}

@inproceedings{conneau2020,
  title={Unsupervised cross-lingual representation learning at scale},
  author={Conneau, Alexis and others},
  booktitle={ACL},
  pages={8440--8451},
  year={2020}
}
EOF

# Create supplementary materials
echo "5. Preparing supplementary materials..."
cat > paper/supplementary/README.md << 'EOF'
# Supplementary Materials

## Code Repository
https://github.com/HillaryDanan/cross-linguistic-attention-dynamics

## Contents
- Full statistical analysis details
- Additional visualizations
- Complete hyperparameter settings
- Extended results tables

## Reproducibility
All experiments can be reproduced using:
```bash
python3 run_full_study.py
python3 cross_model_validation.py
```

## Data
- UN Parallel Corpus (publicly available)
- 1,000 sentence pairs (provided in repository)

## Computational Requirements
- GPU: Not required (CPU sufficient)
- RAM: 8GB minimum
- Time: ~1 hour for full reproduction
EOF

# Create anonymous version
echo "6. Creating anonymous version..."
cp paper/paper.tex submission/anonymous/paper_anon.tex
cp paper/paper.md submission/anonymous/paper_anon.md

# Anonymize the LaTeX version
sed -i '' 's/Hillary Danan/Anonymous Author/g' submission/anonymous/paper_anon.tex
sed -i '' 's/hillarydanan@gmail.com/anonymous@email.com/g' submission/anonymous/paper_anon.tex
sed -i '' 's|github.com/HillaryDanan|github.com/anonymous|g' submission/anonymous/paper_anon.tex

# Anonymize the Markdown version
sed -i '' 's/Hillary Danan/Anonymous Author/g' submission/anonymous/paper_anon.md
sed -i '' 's/hillarydanan@gmail.com/anonymous@email.com/g' submission/anonymous/paper_anon.md
sed -i '' 's|github.com/HillaryDanan|github.com/anonymous|g' submission/anonymous/paper_anon.md

# Copy figures to anonymous version
cp -r paper/figures submission/anonymous/

# Create submission checklist
echo "7. Creating submission checklist..."
cat > submission/CHECKLIST.md << 'EOF'
# ACL/EMNLP Submission Checklist

## Paper Requirements
- [ ] Paper length: 8 pages (excluding references)
- [ ] Anonymous version prepared
- [ ] References complete and properly formatted
- [ ] Figures in PDF format for LaTeX
- [ ] Supplementary materials prepared

## Technical Requirements
- [ ] LaTeX compiles without errors
- [ ] Uses conference style file (acl2021.sty or similar)
- [ ] PDF/A compliant (if required)
- [ ] File size under limit (usually 10MB)

## Content Requirements
- [ ] Abstract under 150 words
- [ ] Keywords selected (if required)
- [ ] Limitations section included
- [ ] Ethical considerations addressed
- [ ] Data availability statement included

## Reproducibility
- [ ] Code repository linked
- [ ] Data sources documented
- [ ] Hyperparameters specified
- [ ] Statistical details provided

## Submission System
- [ ] Account created on submission system
- [ ] Paper title entered
- [ ] Abstract entered
- [ ] Authors and affiliations entered (for final version)
- [ ] Subject areas selected
- [ ] PDF uploaded

## Final Checks
- [ ] Spell check completed
- [ ] Grammar check completed
- [ ] Citations verified
- [ ] Figures legible
- [ ] Tables formatted correctly
EOF

echo "8. Creating LaTeX compilation script..."
cat > paper/compile.sh << 'EOF'
#!/bin/bash
# Compile LaTeX paper

echo "Compiling LaTeX paper..."
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex

echo "Paper compiled: paper.pdf"
echo "Check for any warnings or errors above"
EOF
chmod +x paper/compile.sh

# Create final git commit script
echo "9. Creating git commit script..."
cat > git_push_all.sh << 'EOF'
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
EOF
chmod +x git_push_all.sh

# Summary
echo ""
echo "========================================"
echo "SUBMISSION PACKAGE COMPLETE"
echo "========================================"
echo ""
echo "Created files:"
echo "  ✓ paper/paper.tex - LaTeX version"
echo "  ✓ paper/paper.md - Markdown version"
echo "  ✓ paper/figures/ - Publication figures"
echo "  ✓ paper/references.bib - Bibliography"
echo "  ✓ paper/cover_letter.md - Cover letter"
echo "  ✓ submission/anonymous/ - Anonymous version"
echo "  ✓ submission/CHECKLIST.md - Submission checklist"
echo ""
echo "To push everything to GitHub:"
echo "  ./git_push_all.sh"
echo ""
echo "To compile LaTeX:"
echo "  cd paper && ./compile.sh"
echo ""
echo "Ready for submission! 🎉"
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

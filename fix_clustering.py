import fileinput
import sys

# Read the file and fix the clustering calculation
with open('src/attention_metrics.py', 'r') as f:
    lines = f.readlines()

# Find and replace the import line
for i, line in enumerate(lines):
    if 'from scipy.sparse.csgraph import clustering' in line:
        lines[i] = 'from scipy.sparse import csr_matrix\n'
        break

# Rewrite the file
with open('src/attention_metrics.py', 'w') as f:
    f.writelines(lines)

print("Fixed import issue!")

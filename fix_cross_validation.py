#!/usr/bin/env python3
"""Fix JSON serialization in cross_model_validation.py"""

with open('cross_model_validation.py', 'r') as f:
    content = f.read()

# Add import for numpy at top if not there
if 'import numpy as np' not in content:
    content = content.replace('import json', 'import json\nimport numpy as np')

# Fix the json.dump line to handle numpy types
old_line = '        json.dump(results, f, indent=2)'
new_line = '''        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        json.dump(results, f, indent=2, default=convert_numpy)'''

content = content.replace(old_line, new_line)

# Also fix the significant line
content = content.replace(
    "results['significant'] = p_value < 0.05",
    "results['significant'] = bool(p_value < 0.05)"
)

with open('cross_model_validation.py', 'w') as f:
    f.write(content)

print("Fixed! Run again: python3 cross_model_validation.py")

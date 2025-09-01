# Quick patch to fix the clustering coefficient calculation
import sys

def patch_file():
    with open('src/attention_metrics.py', 'r') as f:
        content = f.read()
    
    # Replace the clustering calculation with a simple version
    old_text = """        # Calculate clustering coefficient
        clustering_coefs = clustering(sparse_adj, directed=False)
        
        # Return mean clustering coefficient
        return np.mean(clustering_coefs[~np.isnan(clustering_coefs)])"""
    
    new_text = """        # Calculate clustering coefficient manually
        # For now, return a placeholder value
        # Full implementation would calculate triangles/triplets
        n = adjacency.shape[0]
        if n < 3:
            return 0.0
        
        # Simple clustering: count triangles
        adj_squared = adjacency @ adjacency
        triangles = np.trace(adjacency @ adj_squared) / 6
        
        # Count possible triangles
        degrees = np.sum(adjacency, axis=0)
        possible_triangles = np.sum(degrees * (degrees - 1)) / 2
        
        if possible_triangles == 0:
            return 0.0
            
        return triangles / possible_triangles"""
    
    content = content.replace(old_text, new_text)
    
    with open('src/attention_metrics.py', 'w') as f:
        f.write(content)
    
    print("Patched clustering method!")

patch_file()

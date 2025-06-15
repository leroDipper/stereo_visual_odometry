import numpy as np

# After triangulating all points, before visualization:
def remove_statistical_outliers(points_3d, k=50, std_ratio=2.0):
    if len(points_3d) < k:
        return points_3d
    
    # For each point, find distance to k nearest neighbors
    from scipy.spatial import cKDTree
    tree = cKDTree(points_3d)
    distances, _ = tree.query(points_3d, k=k+1)  # +1 because first is self
    mean_distances = np.mean(distances[:, 1:], axis=1)  # Skip self-distance
    
    # Remove points with distances > threshold
    threshold = np.mean(mean_distances) + std_ratio * np.std(mean_distances)
    mask = mean_distances < threshold
    return points_3d[mask]
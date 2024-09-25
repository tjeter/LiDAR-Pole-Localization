import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def icp_scipy(source, target, max_iterations=100, tolerance=1e-6):
    for i in range(max_iterations):
        # Find nearest neighbors
        tree = KDTree(target)
        distances, indices = tree.query(source)

        # Extract matched points
        matched_source = source
        matched_target = target[indices]

        # Estimate rigid transformation
        R, t = estimate_rigid_transform(matched_source, matched_target)

        # Apply transformation to source points
        source = np.dot(source, R.T) + t

        # Check convergence
        mean_distance = np.mean(distances)
        if mean_distance < tolerance:
            break

    return source

def estimate_rigid_transform(source, target):
    # Centroid of the source and target
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)

    # Subtract centroids
    source_centered = source - centroid_source
    target_centered = target - centroid_target

    # Compute covariance matrix
    H = np.dot(source_centered.T, target_centered)

    # Singular value decomposition
    U, _, Vt = np.linalg.svd(H)

    # Rotation matrix
    R = np.dot(Vt.T, U.T)

    # Translation vector
    t = centroid_target - np.dot(centroid_source, R.T)

    return R, t

def calculate_precision_recall_f1(global_map_points, aligned_localization_map, threshold=0.5):
    # Assuming that both global_map_points and aligned_localization_map are numpy arrays
    min_points = min(aligned_localization_map.shape[0], global_map_points.shape[0])
    # Binary masks based on some threshold
    global_map_binary = (global_map_points[:min_points] > threshold).astype(int)
    aligned_localization_binary = (aligned_localization_map[:min_points] > threshold).astype(int)

    # True positives, false positives, and false negatives
    tp = np.sum(global_map_binary * aligned_localization_binary)
    fp = np.sum(aligned_localization_binary) - tp
    fn = np.sum(global_map_binary) - tp

    # Precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

# Load global map from file
global_map_data = np.load('globalmap_3.npz')
global_map_points = global_map_data['polemeans'][:, :2]  

# Load localization map from file
localization_map_data = np.load('localization_2023-11-30_23-06-03.npz')
localization_map_points = localization_map_data['T_w_velo_est'][:, :2, 3]  

# Assuming global_map_points and localization_map_points are your point clouds
aligned_localization_map = icp_scipy(localization_map_points[:, :2], global_map_points)

# Ensure both sets have the same number of points
min_points = min(aligned_localization_map.shape[0], global_map_points.shape[0])

mse = np.mean(np.square(aligned_localization_map[:min_points] - global_map_points[:min_points]))
print("MSE:", mse)

rmse = np.sqrt(mse)
print("RMSE:", rmse)

# Calculate precision, recall, and F1 score
precision, recall, f1 = calculate_precision_recall_f1(global_map_points, aligned_localization_map)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Now aligned_localization_map contains the aligned localization points
# Plot the original localization map in blue
plt.scatter(localization_map_points[:, 0], localization_map_points[:, 1], c='blue', label='Original Localization Map', s=3)

# Plot the aligned localization map in red
plt.scatter(aligned_localization_map[:, 0], aligned_localization_map[:, 1], c='red', label='Aligned Localization Map', s=3)

# Plot the global map in green
plt.scatter(global_map_points[:, 0], global_map_points[:, 1], c='green', label='Global Map', s=5)

plt.legend()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Original, Aligned Localization, and Global Maps')
plt.show()


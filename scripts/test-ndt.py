import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

def initialize_ndt_grid(points, grid_size):
    # Compute the grid indices for each point
    grid_indices = np.floor(points[:, :2] / grid_size).astype(int)
    
    # Initialize an empty dictionary to store normal distributions for each grid cell
    ndt_grid = {}
    
    for i in range(len(points)):
        grid_index = tuple(grid_indices[i])
        if grid_index not in ndt_grid:
            ndt_grid[grid_index] = [points[i, :]]
        else:
            ndt_grid[grid_index].append(points[i, :])
    
    # Compute normal distributions for each grid cell
    for key, value in ndt_grid.items():
        mean = np.mean(value, axis=0)
        cov = np.cov(value, rowvar=False)
        ndt_grid[key] = multivariate_normal(mean, cov, allow_singular=True)
    
    return ndt_grid

def ndt_cost_function(transform, local_points, global_ndt_grid, grid_size):
    # Reshape transform to a 2D array
    transform = transform.reshape((3, 3))
    
    # Apply the transformation to the local points
    transformed_local_points = np.dot(transform[:2, :2], local_points[:, :2].T).T + transform[:2, 2]

    # Reshape transformed_local_points if needed
    transformed_local_points = transformed_local_points.reshape(-1, 2)

    # Compute the cost by comparing the local points with global NDT distributions
    cost = 0.0
    for i in range(len(local_points)):
        grid_index = tuple(np.floor(transformed_local_points[i] / grid_size).astype(int))
        if grid_index in global_ndt_grid:
            cost += -np.log(global_ndt_grid[grid_index].pdf(local_points[i]))

    return cost


#def ndt_cost_function(transform, local_points, global_ndt_grid, grid_size):
#    # Apply the transformation to the local points
#    transformed_local_points = np.dot(transform[:2, :2], local_points[:, :2].T).T + transform[:2, 2]
#
#    # Reshape transformed_local_points if needed
#    transformed_local_points = transformed_local_points.reshape(-1, 2)
#
#    # Compute the cost by comparing the local points with global NDT distributions
#    cost = 0.0
#    for i in range(len(local_points)):
#        grid_index = tuple(np.floor(transformed_local_points[i] / grid_size).astype(int))
#        if grid_index in global_ndt_grid:
#            cost += -np.log(global_ndt_grid[grid_index].pdf(local_points[i]))
#
#    return cost


#def ndt_cost_function(transform, local_points, global_ndt_grid):
#    # Apply the transformation to the local points
#    transformed_local_points = np.dot(transform[:2, :2], local_points[:, :2].T).T + transform[:2, 2]
#    
#    # Compute the cost by comparing the local points with global NDT distributions
#    cost = 0.0
#    for i in range(len(local_points)):
#        grid_index = tuple(np.floor(transformed_local_points[i] / grid_size).astype(int))
#        if grid_index in global_ndt_grid:
#            cost += -np.log(global_ndt_grid[grid_index].pdf(local_points[i]))
#    
#    return cost

def ndt_registration(local_points, global_ndt_grid, grid_size, initial_guess=None):
    if initial_guess is None:
        initial_guess = np.eye(3)
    
    result = minimize(ndt_cost_function, initial_guess, args=(local_points, global_ndt_grid, grid_size), method='BFGS')
    
    return result.x

def visualize_registration(local_points, transformed_local_points, global_points):
    plt.figure(figsize=(8, 8))
    
    plt.scatter(global_points[:, 0], global_points[:, 1], c='blue', label='Global Map')
    plt.scatter(local_points[:, 0], local_points[:, 1], c='green', label='Original Local Map')
    plt.scatter(transformed_local_points[:, 0], transformed_local_points[:, 1], c='red', label='Transformed Local Map')
    
    plt.title('NDT Registration Result')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_registration(local_points, transformed_local_points, global_points):
    # Define ground truth correspondences (modify based on your actual ground truth)
    ground_truth_indices = np.arange(len(global_points))

    # Find predicted correspondences (simple nearest neighbors in this example)
    predicted_indices = find_predicted_correspondences(local_points, transformed_local_points, global_points)

    # Compute precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth_indices, predicted_indices, average='binary')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

def find_predicted_correspondences(local_points, transformed_local_points, global_points):
    # Simple nearest neighbors
    distances = np.linalg.norm(transformed_local_points[:, :2] - global_points[:, :2], axis=1)
    predicted_indices = np.argmin(distances, axis=0)
    return predicted_indices

# Example usage:
# Assume local_points and global_points are your local and global point clouds
grid_size = 1.0

# Load global map from file
global_map_data = np.load('globalmap_3.npz')
global_map_points = global_map_data['polemeans'][:, :2] 

# Load localization map from file
localization_map_data = np.load('localization_2023-11-30_23-06-03.npz')
localization_map_points = localization_map_data['T_w_velo_est'][:, :2, 3] 

local_ndt_grid = initialize_ndt_grid(localization_map_points, grid_size)
global_ndt_grid = initialize_ndt_grid(global_map_points, grid_size)

# Perform NDT registration
registration_result = ndt_registration(localization_map_points, global_ndt_grid, grid_size)

# Apply the transformation to the local points
transformed_local_points = np.dot(registration_result[:2, :2], localization_map_points[:, :2].T).T + registration_result[:2, 2]

# Visualize the registration result
visualize_registration(localization_map_points, transformed_local_points, global_points)

# Evaluate precision, recall, and F1-score
evaluate_registration(localization_map_points, transformed_local_points, global_points)


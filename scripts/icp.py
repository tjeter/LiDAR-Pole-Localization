import open3d as o3d
import numpy as np

# Load your global map and estimated localization point cloud
global_map_file = 'globalmap_3.npz'  
localization_estimation_file = 'localization_2023-11-30_23-06-03.npz'  

# Load global map and estimated localization point cloud
with np.load(global_map_file, allow_pickle=True) as data:
    global_map_points = data['polemeans'][:, :2]  
print(global_map_points.shape)
print(type(global_map_points))
with np.load(localization_estimation_file, allow_pickle=True) as data:
    localization_estimation_points = data['T_w_velo_est'][:, :2, 3]  

# Convert the point clouds to open3d format
global_map_cloud = o3d.geometry.PointCloud()
global_map_cloud.points = o3d.utility.Vector3dVector(global_map_points)

localization_estimation_cloud = o3d.geometry.PointCloud()
localization_estimation_cloud.points = o3d.utility.Vector3dVector(localization_estimation_points)

# Perform ICP registration
threshold = 0.2  # Adjust the threshold based on your data
trans_init = np.eye(4)  # Initial transformation matrix
reg_p2p = o3d.registration.registration_icp(
    localization_estimation_cloud, global_map_cloud, threshold, trans_init,
    o3d.registration.TransformationEstimationPointToPoint(),
    o3d.registration.ICPConvergenceCriteria(max_iteration=2000))

# Get the transformation matrix
T_localization_to_global = reg_p2p.transformation
print("Transformation matrix:")
print(T_localization_to_global)

# Transform the estimated localization points to the global map coordinates
transformed_localization_points = np.dot(T_localization_to_global[:3, :3], localization_estimation_points.T).T + T_localization_to_global[:3, 3]

# Visualize the results
o3d.visualization.draw_geometries([global_map_cloud, localization_estimation_cloud.transform(T_localization_to_global)])


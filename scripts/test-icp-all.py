import open3d as o3d
import numpy as np
import pandas as pd

# Load your global map
global_map_file = 'globalmap_3.npz'  

# Load global map point cloud
with np.load(global_map_file, allow_pickle=True) as data:
    global_map_points = data['polemeans'][:, :2]  

# Add a third column for Z-coordinate (zeros in this case)
global_map_points_3d = np.hstack([global_map_points, np.zeros((global_map_points.shape[0], 1))])

# Create a Pandas DataFrame
global_df = pd.DataFrame(global_map_points_3d, columns=['x', 'y', 'z'])

# Create an Open3D point cloud for the global map
global_cloud = o3d.geometry.PointCloud()
global_cloud.points = o3d.utility.Vector3dVector(global_df[['x', 'y', 'z']].values)

# Load your localization map
localization_map_file = 'localization_2023-11-30_23-06-03.npz'  

# Load localization map point cloud
with np.load(localization_map_file, allow_pickle=True) as data:
    localization_map_points = data['T_w_velo_est'][:, :2, 3]  

# Add a third column for Z-coordinate (zeros in this case)
localization_map_points_3d = np.hstack([localization_map_points, np.zeros((localization_map_points.shape[0], 1))])

# Create a Pandas DataFrame
localization_df = pd.DataFrame(localization_map_points_3d, columns=['x', 'y', 'z'])

# Create an Open3D point cloud for the localization map
localization_cloud = o3d.geometry.PointCloud()
localization_cloud.points = o3d.utility.Vector3dVector(localization_df[['x', 'y', 'z']].values)

# Set the color of the localization_cloud to red
localization_cloud.paint_uniform_color([1, 0, 0])  # [R, G, B], 1 for red, 0 for green, 0 for blue

# Perform visualization with both point clouds
o3d.visualization.draw_geometries([global_cloud, localization_cloud])


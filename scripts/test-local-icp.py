import open3d as o3d
import numpy as np
import pandas as pd

# Load your localization map
localization_map_file = 'localization_2023-11-30_23-06-03.npz'  # Replace with the actual file path

# Load localization map point cloud
with np.load(localization_map_file, allow_pickle=True) as data:
    localization_map_points = data['T_w_velo_est'][:, :2, 3]  # Assuming 'T_w_velo_est' contains the localization poses

# Add a third column for Z-coordinate (zeros in this case)
localization_map_points_3d = np.hstack([localization_map_points, np.zeros((localization_map_points.shape[0], 1))])

# Create a Pandas DataFrame
localization_df = pd.DataFrame(localization_map_points_3d, columns=['x', 'y', 'z'])

# Create an Open3D point cloud for the localization map
localization_cloud = o3d.geometry.PointCloud()
localization_cloud.points = o3d.utility.Vector3dVector(localization_df[['x', 'y', 'z']].values)

# Perform visualization
o3d.visualization.draw_geometries([localization_cloud])

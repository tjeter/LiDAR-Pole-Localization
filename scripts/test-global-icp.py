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
df = pd.DataFrame(global_map_points_3d, columns=['x', 'y', 'z'])

# Create an Open3D point cloud from the DataFrame
global_map_cloud = o3d.geometry.PointCloud()
global_map_cloud.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].values)

# Perform visualization
o3d.visualization.draw_geometries([global_map_cloud])


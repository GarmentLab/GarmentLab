import os
import shutil
import numpy as np

directory = 'data/with_sleeves/cd_original/mesh_pcd'

paths_with_points_out_of_range = []

# Traverse directory and process npz files
for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith('.npz'):
            file_path = os.path.join(root, file)
            try:
                data = np.load(file_path)
                if 'mesh_points' in data:
                    pcd_points = data['mesh_points']
                    # Check for points outside the specified ranges
                    x_out_of_range = np.any((pcd_points[:, 0] < -3) | (pcd_points[:, 0] > 3))
                    y_out_of_range = np.any((pcd_points[:, 1] < 0) | (pcd_points[:, 1] > 4))
                    if x_out_of_range or y_out_of_range:
                        paths_with_points_out_of_range.append(file_path)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Filter and delete only specific file's directory
# for path in paths_with_points_out_of_range:
#     if path.endswith('p_3.npz'):  # Adjust this condition to match the specific file
#         folder_to_delete = os.path.dirname(path)
#         try:
#             # shutil.rmtree(folder_to_delete)  # Delete the directory
#             print(f"Deleted folder: {folder_to_delete}")
#         except Exception as e:
#             print(f"Error deleting folder {folder_to_delete}: {e}")
            

print(len(paths_with_points_out_of_range))         
for path in paths_with_points_out_of_range:
    print(path)
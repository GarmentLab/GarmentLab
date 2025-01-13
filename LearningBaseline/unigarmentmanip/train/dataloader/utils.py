import os
import open3d as o3d
import numpy as np

def get_deformation_paths(root_dir, garment_data_num, train_ratio, mode: str=None):

    deformation_paths = []
    
    # 遍历根目录中的每个文件夹
    for folder_name in sorted(os.listdir(root_dir)):

        folder_path = os.path.join(root_dir, folder_name)
        
        if os.path.isdir(folder_path): # and 'Dress' not in folder_name:
            
            # 获取该文件夹中所有 .npz 文件路径
            paths = [os.path.join(folder_path, file) 
                        for file in sorted(os.listdir(folder_path)) 
                        if file.endswith('.npz')]
            if paths:  
                
                paths = sorted(paths, key=lambda path: int(path.split('/')[-1].split('.')[0].split('_')[-1]))
                print(len(paths))
                deformation_paths.append(paths)

    deformation_paths = sorted(deformation_paths, key=lambda paths: int(paths[0].split('/')[-2].split('_')[0]))
    print(len(deformation_paths))
    
    if garment_data_num is not None:
        deformation_paths = deformation_paths[:garment_data_num]
        
    split_idx = int(len(deformation_paths) * train_ratio)
    
    if mode is None:
        return deformation_paths
    
    if mode == 'train':
        return deformation_paths[:split_idx]
    elif mode == 'val':
        return deformation_paths[split_idx:]
    else:
        raise ValueError("Invalid mode. Please choose 'train' or 'val'.")



def create_cross_deformation_pairs(all_deform_paths):
    """
    构建跨变形配对：同一件衣服的不同变形状态之间的配对。
    
    Args:
        all_deform_paths (list[list[str]]): 每个子列表包含一件衣服的所有变形路径。
    
    Returns:
        list[tuple[str, str]]: 每对元素是同一件衣服的不同变形状态文件路径配对。
    """ 
    deformation_pairs = []
    
    # 遍历每一件衣服的变形路径
    for deform_paths in all_deform_paths:
        # 对每件衣服的变形路径，两两组合成配对
        for i in range(len(deform_paths)):
            for j in range(i + 1, len(deform_paths)):
                deformation_pairs.append((deform_paths[i], deform_paths[j]))
    
    return deformation_pairs


def create_flat2deform_pairs(all_deform_paths):
    """
    构建跨变形配对：同一件衣服的不同变形状态之间的配对。
    
    Args:
        all_deform_paths (list[list[str]]): 每个子列表包含一件衣服的所有变形路径。
    
    Returns:
        list[tuple[str, str]]: 每对元素是同一件衣服的不同变形状态文件路径配对。
    """
    deformation_pairs = []
    
    # 遍历每一件衣服的变形路径
    for deform_paths in all_deform_paths:
        # 对每件衣服的变形路径，0和i配对
        for i in range(1, len(deform_paths)):
            deformation_pairs.append((deform_paths[0], deform_paths[i]))
    
    return deformation_pairs
    
def create_cross_object_pairs(all_deform_paths):
    """
    构建跨物体配对：不同衣服之间的变形状态配对。
    
    Args:
        all_deform_paths (list[list[str]]): 每个子列表包含一件衣服的所有变形路径。
    
    Returns:
        list[tuple[str, str]]: 每对元素是不同衣服的变形状态文件路径配对。
    """
    object_pairs = []
    
    # 遍历每一对不同衣服的组合
    for i in range(len(all_deform_paths)):
        for j in range(i + 1, len(all_deform_paths)):
            # 将衣服 i 和衣服 j 的所有变形路径两两配对
            
            # for m in range(len(all_deform_paths[i])):
            #     for idx in range(5):
            #         object_pairs.append((all_deform_paths[i][m], all_deform_paths[j][idx]))  
            # for n in range(len(all_deform_paths[j])):
            #     for idx in range(5):
            #         object_pairs.append((all_deform_paths[i][idx], all_deform_paths[j][n]))   
            
            for m in range(len(all_deform_paths[i])):
                object_pairs.append((all_deform_paths[i][m], all_deform_paths[j][0]))  
            for n in range(len(all_deform_paths[j])):
                object_pairs.append((all_deform_paths[i][0], all_deform_paths[j][n]))   
            
            # for m in range(len(all_deform_paths[i])):
            #     for n in range(len(all_deform_paths[j])):
            #         object_pairs.append((all_deform_paths[i][m], all_deform_paths[j][n]))  
    
    return object_pairs

def create_cross_only_deformation_pairs(all_deform_paths):
    deformation_pairs = []
    
    # 遍历每一件衣服的变形路径
    for deform_paths in all_deform_paths:
        # 对每件衣服的变形路径，两两组合成配对
        for i in range(1, len(deform_paths)):
            for j in range(i + 1, len(deform_paths)):
                deformation_pairs.append((deform_paths[i], deform_paths[j]))
    
    flat2deform_pairs = []
    for deform_paths in all_deform_paths:
        for j in range(1, len(deform_paths)):
            flat2deform_pairs.append((deform_paths[0], deform_paths[j]))

    # 计算 deformation_pairs 的长度
    len_deformation_pairs = len(deformation_pairs)
    
    # 扩充 flat2deform_pairs 到与 deformation_pairs 相同的长度
    if len(flat2deform_pairs) < len_deformation_pairs:
        # 计算需要重复的次数
        repeat_times = len_deformation_pairs // len(flat2deform_pairs)
        remainder = len_deformation_pairs % len(flat2deform_pairs)
        
        # 重复 flat2deform_pairs
        flat2deform_pairs = flat2deform_pairs * repeat_times + flat2deform_pairs[:remainder]
    
    # 合并两个列表
    combined_pairs = deformation_pairs + flat2deform_pairs
    
    return combined_pairs
    
    

def visualize_point_cloud(pc, correspondence_indices, title="Point Cloud"):
    """
    可视化单个点云，其中对应点以红色显示，其他点为黑色。
    
    Args:
        pc (np.ndarray): 点云, shape (n, 3)
        correspondence_indices (list[int]): 对应点的索引列表
        title (str): 窗口标题
    """
    # 创建 Open3D 点云对象
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc)
    
    colors = np.zeros_like(pc) 
    colors[:, 0] = 0
    colors[:, 1] = 255
    colors[:, 2] = 0
    for idx in correspondence_indices:
        colors[idx] = [1, 0, 0]  # 红色
    
    cloud.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"Visualizing {title}...")
    o3d.visualization.draw_geometries([cloud])

def fps_with_selected(points, selectable_id, selected_points, k):
    """
    从剩余点中选择 k 个均匀分布的点，确保这些点与已选点之间的距离最大。
    
    points: 原始点云，形状为 (N, 3)
    selectable_id: 可选点的索引列表，形状为 (S,) -> 选择这些索引中的点
    selected_points: 已选的点的坐标列表，形状为 (M, 3)
    k: 要选择的新点的数量
    
    返回：选中的点的坐标和对应的索引
    """
    
    points = np.array(points)
    selected_points = np.array(selected_points)
    
    # 获取可选点的坐标
    selectable_points = points[selectable_id]
    N = selectable_points.shape[0]
    
    # 保存每个点到已选点集合的最小距离
    min_distances = np.full(N, np.inf)
    
    # 计算所有可选点到已选点的距离
    for sp in selected_points:
        dist = np.linalg.norm(selectable_points - sp, axis=1)
        min_distances = np.minimum(min_distances, dist)  # 更新最小距离
    
    # 选择距离已选点最远的k个点
    new_selected_points = []
    for _ in range(k):
        # 找到距离最远的可选点
        farthest_point_idx = np.argmax(min_distances)
        new_selected_points.append(farthest_point_idx)
        
        # 更新到新选点的最小距离
        dist = np.linalg.norm(selectable_points - selectable_points[farthest_point_idx], axis=1)
        min_distances = np.minimum(min_distances, dist)  # 更新最小距离

    # 获取选中的点的坐标和对应的原始点云的索引
    selected_coords = selectable_points[new_selected_points]  # 获取选中的点的坐标
    selected_indices = selectable_id[new_selected_points]  # 获取对应的原始索引

    return selected_indices, selected_coords  # 返回索引和坐标

def nearest_mesh2pcd(mesh_points, pcd_points, mesh_keypoints_id):
                   
    from scipy.spatial import cKDTree

    mesh_keypoints = mesh_points[mesh_keypoints_id]

    pcd_tree = cKDTree(pcd_points)

    _, nearest_indices = pcd_tree.query(mesh_keypoints)

    return nearest_indices.tolist()

def nearest_pcd2mesh(pcd_points, mesh_points, pcd_keypoints_id):
                   
    from scipy.spatial import cKDTree

    pcd_keypoints = pcd_points[pcd_keypoints_id]

    mesh_tree = cKDTree(mesh_points)

    _, nearest_indices = mesh_tree.query(pcd_keypoints)

    return nearest_indices.tolist()

if __name__ == '__main__':
    root_dir = 'dress_data/cross_deformation'
    deformation_paths = get_deformation_paths(root_dir, None, 0.8, 'train')
    
    print(len(deformation_paths))
    
    deformation_pairs = create_cross_deformation_pairs(deformation_paths)
    object_pairs = create_cross_object_pairs(deformation_paths)
    # print(object_pairs[1])
    print(len(deformation_pairs))
    print(len(object_pairs))
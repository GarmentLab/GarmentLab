import os
import numpy as np

def normalize_points(pcd_points): 
    """
    Normalize point cloud to center at the origin.
    """
    # 计算质心
    centroid = np.mean(np.asarray(pcd_points), axis=0)
    # 将点云平移到原点
    normalized_points = np.asarray(pcd_points) - centroid
    return normalized_points, centroid

def read_obj_vertices(obj_path):
    """
    Read vertex data from an .obj file.
    """
    vertices = []
    with open(obj_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # 只读取顶点信息
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def save_normalized_points(normalized_points, save_path):
    """
    Save normalized points to a .npz file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目标目录存在
    np.savez(save_path, normalized_points=normalized_points)
    print(f"Saved normalized points to {save_path}")

def process_obj_paths(input_file, output_base_path):
    """
    Process each .obj file path in the input file, normalize the points, and save as .npz.
    """
    with open(input_file, 'r') as file:
        obj_paths = file.read().splitlines()
    
    for i, obj_path in enumerate(obj_paths):
        # 读取点云数据
        try:
            vertices = read_obj_vertices(obj_path)
            normalized_points, _ = normalize_points(vertices)
        except Exception as e:
            print(f"Error processing {obj_path}: {e}")
            continue

        # 生成保存路径
        relative_path = obj_path.split('/')[-1].split('.')[0] + '_obj'
        
        relative_path = f"{i}_{relative_path}"
        print(relative_path)
        save_dir = os.path.join(output_base_path, relative_path)
        print(save_dir)
        save_path = os.path.join(save_dir, f"p_0.npz")
        print(save_path)
        
        # 保存点云数据
        save_normalized_points(normalized_points, save_path)


if __name__ == "__main__":

    # 输入和输出路径
    input_file = "unigarment/collect/collect_cd/prepare/filtered_obj_paths.txt"
    output_base_path = "data/with_sleeves/cd_processed/bulgy_mesh"

    process_obj_paths(input_file, output_base_path)



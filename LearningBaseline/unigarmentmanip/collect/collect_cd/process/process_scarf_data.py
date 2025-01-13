import os 
import sys
sys.path.append(os.getcwd())
sys.path.append("/home/user/DexGarmentLab-master/DexGarmentLab-master/unigarment/unigarmentmlp")
sys.path.append("/home/user/DexGarmentLab-master/DexGarmentLab-master/unigarment/unigarmentmlp/merger")
sys.path.append("/home/user/DexGarmentLab-master/DexGarmentLab-master/unigarment")
import numpy as np

from unigarment.collect.pcd_utils import normalize_pcd_points, get_visible_indices, nearest_mesh2pcd, rotate_point_cloud
from unigarment.unigarmentmlp.predict import get_skeleton
from collect.design_skeleton.add_scarf_skeleton import get_scarf_mesh_skeleton_id
from tqdm import tqdm

def modify_pcd(input_dir, output_dir):
    
    for subdir, _, files in os.walk(input_dir):
        relative_subdir = os.path.relpath(subdir, input_dir)
        target_subdir = os.path.join(output_dir, relative_subdir)

        # 创建目标目录
        os.makedirs(target_subdir, exist_ok=True)

        # 用 scale 标准化其余文件
        for file in files:
            if file.endswith('.npz'):
                file_path = os.path.join(subdir, file)
                data = np.load(file_path)

                mesh_points = data['mesh_points']
                mesh_visible_indices = get_visible_indices(mesh_points)
                pcd_points = mesh_points[mesh_visible_indices]

                # 保存到目标目录
                np.savez(os.path.join(target_subdir, file), 
                        pcd_points=pcd_points, 
                        mesh_points=mesh_points)
             

# normalize
def normalize_mesh_and_pcd(input_dir, output_dir):

    normalized_garment = 0
    
    for subdir, _, files in os.walk(input_dir):
        relative_subdir = os.path.relpath(subdir, input_dir)
        target_subdir = os.path.join(output_dir, relative_subdir)

        # 创建目标目录
        os.makedirs(target_subdir, exist_ok=True)

        # 获取 p_0.npz 的路径
        p_0_path = os.path.join(subdir, 'p_0.npz')
        if not os.path.exists(p_0_path):
            continue

        # 使用 p_0.npz 计算 centroid 和 scale
        p_0_data = np.load(p_0_path)
        pcd_points = p_0_data['pcd_points']
        normalized_p_0_points, flat_centroid, flat_scale = normalize_pcd_points(pcd_points)

        # 对 mesh_points 也进行相同归一化
        mesh_points = p_0_data['mesh_points']
        normalized_mesh_points = (mesh_points - flat_centroid) * flat_scale

        # 保存标准化的 p_0.npz
        np.savez(os.path.join(target_subdir, 'p_0.npz'), 
                pcd_points=normalized_p_0_points, 
                mesh_points=normalized_mesh_points,
                visible_mesh_indices=get_visible_indices(mesh_points))

        # 用 scale 标准化其余文件
        for file in files:
            if file.endswith('.npz') and file != 'p_0.npz':
                file_path = os.path.join(subdir, file)
                data = np.load(file_path)

                # 使用 p_0.npz 的参数进行标准化
                pcd_points = data['pcd_points']
                centroid = np.mean(pcd_points, axis=0)
                normalized_pcd_points = (pcd_points - centroid) * flat_scale

                mesh_points = data['mesh_points']
                normalized_mesh_points = (mesh_points - centroid) * flat_scale

                # 保存到目标目录
                np.savez(os.path.join(target_subdir, file), 
                        pcd_points=normalized_pcd_points, 
                        mesh_points=normalized_mesh_points,
                        visible_mesh_indices=get_visible_indices(mesh_points))
        
        normalized_garment += 1
        print(f"Normalized {normalized_garment} garments")       




def process_info(dir):
    for subdir, _, files in os.walk(dir):
        
        for file in files:
            
            # 获取flat mesh_points 从而获取关键点索引
            if file.endswith('p_0.npz'):
                
                flat_cloth_path = os.path.join(subdir, file)
                flat_cloth_points = np.load(flat_cloth_path)['mesh_points']
                
                #----------------------------------------------------------------------
                # skeleton_checkpoint_path = "/home/user/DexGarmentLab-master/DexGarmentLab-master/unigarment/unigarmentmlp/with_sleeves.pth"
                # _, flat_cloth_keypoints_id = get_skeleton(flat_cloth_points, skeleton_checkpoint_path)
                
                ######################## 一定要修改啊啊啊啊啊啊啊 ####################！！！！！！！！！！！！！！！！
                flat_cloth_keypoints_id = get_scarf_mesh_skeleton_id(flat_cloth_points)
                ######################## 一定要修改啊啊啊啊啊啊啊 ####################！！！！！！！！！！！！！！！！！
                # ------------------------------------,m------------------------------------
        
        for file in files:
            if file.endswith('.npz'):
                file_path = os.path.join(subdir, file)
                data = np.load(file_path)
                
                mesh_points = data['mesh_points']
                pcd_points = data['pcd_points']
                visible_mesh_indices = data['visible_mesh_indices']
                
                # 衣服mesh的关键点索引(和flat状态一致)
                mesh_cloth_keypoints_id = flat_cloth_keypoints_id
                
                # 衣服camera拍摄的点云中的关键点索引
                pcd_cloth_keypoints_id = nearest_mesh2pcd(mesh_points, pcd_points, mesh_cloth_keypoints_id)

                # 创建关键点是否可见掩码
                keypoints_visible_mask = [
                    1 if keypoint in visible_mesh_indices else 0 
                    for keypoint in mesh_cloth_keypoints_id
                ]
                
                np.savez(file_path,
                        mesh_points=mesh_points,
                        pcd_points=pcd_points,
                        pcd_keypoints_id=pcd_cloth_keypoints_id,
                        mesh_keypoints_id=mesh_cloth_keypoints_id,
                        visible_mesh_indices=visible_mesh_indices,
                        keypoints_visible_mask=keypoints_visible_mask)


                                
                
        

if __name__ == "__main__":
    
    # path = "data/with_sleeves/cd_processed/mesh_pcd/0_DLLS_Dress206_obj/p_0.npz"
    # points = np.load(path)['mesh_points']
    # points, *_ = normalize_pcd_points(points)
    # id = get_bulgy_cloth_keypoints_id(points)
    # print(id)
    
    type = "scarf"
    input_dir = f"data/{type}/cd_original/mesh_pcd"
    output_dir = f"data/{type}/cd_processed/mesh_pcd"
    os.makedirs(output_dir, exist_ok=True)
    
    normalize_mesh_and_pcd(input_dir, output_dir)
    
    process_info(output_dir)
    
    
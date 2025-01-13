import open3d as o3d
import os

def find_obj_files(base_path):
    obj_files = []
    for path in base_path:
        for root, dirs, files in os.walk(path):
            for file in files:
                
                if 'Tie' in file:
                    continue
                    
                if file.endswith(".obj") and file != "border.obj":
                    full_path = os.path.join(root, file)
                    obj_files.append(full_path)
                    
    return obj_files


def read_and_visualize_obj(filepath):
    # 读取 .obj 文件
    mesh = o3d.io.read_triangle_mesh(filepath)
    
    # 获取顶点作为点云
    point_cloud = mesh.sample_points_uniformly(number_of_points=5000)  # 可调整点数量
    
    # 计算点云的中心
    center = point_cloud.get_center()
    
    # 创建坐标轴并平移到点云的中心
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # size 可以调整坐标轴的大小
    coordinate_frame.translate(center)  # 平移坐标轴到点云中心
    
    # 可视化点云和坐标轴
    o3d.visualization.draw_geometries([point_cloud, coordinate_frame],
                                      window_name="Point Cloud with Centered Coordinate Axes",
                                      width=800,
                                      height=600,
                                      point_show_normal=False)



def save_path_to_file(output_dir, category, path):
    file_path = os.path.join(output_dir, f"{category}.txt")
    with open(file_path, 'a') as f:  # 使用 'a' 模式，每次都追加到文件末尾
        f.write(path + '\n')
    print(f"Saved {path} to {category}.txt")

def save_obj_files_to_txt(paths, output_file):

    with open(output_file, 'w') as f:
        for path in paths:
            f.write(path + '\n')

    print(f"路径已写入 {output_file}")
    

if __name__ == "__main__":
    
    # type = "ls_tops"
    # output_dir = f"unigarment/collect/collect_skeleton_data/obj_paths/{type}"
    # os.makedirs(output_dir, exist_ok=True)

    # 获取 .obj 文件路径
    # garment_dir = ["Assets/Garment/Dress/Long_LongSleeve",
    #                "Assets/Garment/Dress/Long_ShortSleeve",
    #                "Assets/Garment/Dress/Short_LongSleeve",
    #                "Assets/Garment/Dress/Short_ShortSleeve",
    #                "Assets/Garment/Tops/Collar_Lsleeve_FrontClose",
    #                "Assets/Garment/Tops/Collar_Ssleeve_FrontClose",
    #                "Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose",
    #                "Assets/Garment/Tops/NoCollar_Ssleeve_FrontClose"
    #                ]

    # garment_dir = ["Assets/Garment/Trousers/Short"]
    
    # paths = find_obj_files(garment_dir)
    # print(f"Found {len(paths)} .obj files in {garment_dir}")
    # save_obj_files_to_txt(paths, os.path.join(output_dir, f"{type}.txt"))
    
    # file_path = f"unigarment/collect/collect_skeleton_data/obj_paths/{type}/{type}.txt"
    file_path = "unigarment/collect/collect_skeleton_data/obj_paths/front_open/front_open.txt"
    with open(file_path, 'r') as f:
        obj_paths = f.readlines()
    
    for path in obj_paths:
        print(path.strip())
        read_and_visualize_obj(path.strip())
    
    # file_path = "unigarment/collect/collect_skeleton_data/obj_paths/hat/duck_hat.txt"
    # with open(file_path, 'r') as f:
    #     obj_paths = f.readlines()
    
    # for path in obj_paths:
    #     print(path.strip())
    #     read_and_visualize_obj(path.strip())
    
    # # 用于保存每个类别的路径
    # categories = {
    #     'category_1': [],
    #     'category_2': [],
    #     'category_3': []
    # }

    # garment_dir = "Assets/Garment/Hat"
    # output_dir = "unigarment/collect/collect_skeleton_data/obj_paths/glove"
    # os.makedirs(output_dir, exist_ok=True)

    # # 获取 .obj 文件路径
    # garment_dir = "Assets/Garment/Glove"
    # paths = find_obj_files(garment_dir)
    # print(f"Found {len(paths)} .obj files in {garment_dir}")
    # save_obj_files_to_txt(paths, os.path.join(output_dir, "glove_all.txt"))
    # qualified_paths = []
    # for i, path in enumerate(paths):
    #     print(i + 1, path)
    #     read_and_visualize_obj(path)
        # user_input = input("1: qualified, 0: not qualified").lower()
        # if user_input == '1':
        #     qualified_paths.append(path)
        # elif user_input == '0':
        #     continue
        # else:
        #     print("Invalid input, please enter 1 or 0")


    
    # for obj_file in obj_paths:
    #     print(f"Processing {obj_file}")
    #     read_and_visualize_obj(obj_file.strip())
        
    #     # 获取用户输入进行分类
    #     user_input = input("Press Enter to continue or 'q' to quit: ").lower()

    #     if user_input == 'q':
    #         print("Exiting...")
    #         break
    #     elif user_input == '1': 
    #         categories['category_1'].append(obj_file)
    #         save_path_to_file(output_dir, 'category_1', obj_file)  # 即时保存
    #     elif user_input == '2':  
    #         categories['category_2'].append(obj_file)
    #         save_path_to_file(output_dir, 'category_2', obj_file)  # 即时保存
    #     elif user_input == '3':  
    #         categories['category_3'].append(obj_file)
    #         save_path_to_file(output_dir, 'category_3', obj_file)  # 即时保存
    #     else:
    #         print("Invalid input, skipping this file.")

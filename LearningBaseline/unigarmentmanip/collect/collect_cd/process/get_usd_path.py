import open3d as o3d
import os

def find_dress_obj_files(base_path):
    dress_obj_files = []
    # 遍历指定目录及其子目录
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # 筛选 .obj 文件并且文件名包含 "Dress"
            if file.endswith(".usd") and file != "border_obj.usd":
                # 拼接完整路径
                full_path = os.path.join(root, file)
                dress_obj_files.append(full_path)
    return dress_obj_files

def save_paths_to_txt(file_paths, output_file):
    # 将路径写入到 txt 文件中
    with open(output_file, 'w') as f:
        for path in file_paths:
            f.write(path + '\n')
    print(f"File paths saved to {output_file}")
    
    
def process_obj_paths(input_file, output_file):

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 移除行尾的换行符
            usd_path = line.strip()
            
            # 如果是空行，跳过
            if not usd_path:
                continue

            # 替换路径中的 ".usd" 为 ".obj"
            obj_path = usd_path.replace('.obj', '_obj.usd')

            # 写入到输出文件
            outfile.write(obj_path + '\n')

    print(f"转换完成！已生成新文件：{output_file}")



if __name__ == "__main__":
    
    # type = "with_sleeves"
    # paths = ["Assets/Garment/Dress/Long_LongSleeve",
    #         "Assets/Garment/Dress/Short_LongSleeve",
    #         "Assets/Garment/Tops/Collar_Lsleeve_FrontClose",
    #         "Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose"]
    
    # output_dir = f"unigarment/collect/collect_cd/prepare/{type}"
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, "usd_paths.txt")
    
    # qualified_paths = []
    
    # for path in paths:       
    #     qualified_path = find_dress_obj_files(path)
    #     qualified_paths.extend(qualified_path)
    
    # save_paths_to_txt(qualified_paths, output_file)   
    # process_obj_paths(output_file, os.path.join(output_dir, "obj_paths.txt"))
    path = "unigarment/collect/collect_cd/prepare/ls_tops/ls_tops.txt"
    process_obj_paths(path, path)


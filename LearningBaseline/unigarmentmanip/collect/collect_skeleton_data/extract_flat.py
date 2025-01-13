import os

# 设置目标文件夹路径
mesh_pcd_path = "mesh_pcd"  # 请将此路径替换为实际路径

# 遍历文件夹及其子文件夹
for root, dirs, files in os.walk(mesh_pcd_path):
    for file in files:
        # 如果是 npz 文件
        if file.endswith(".npz"):
            file_path = os.path.join(root, file)
            
            # 如果文件不是 p_0.npz，删除
            if file != "p_0.npz":
                os.remove(file_path)
                print(f"Deleted: {file_path}")

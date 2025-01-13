import os

def transform_obj_to_usd_txt(obj_path, usd_path):

    with open(obj_path, 'r') as obj_file:
        obj_paths = obj_file.readlines()
    
    # 转换路径
    usd_paths = []
    for obj_file_path in obj_paths:

        obj_file_path = obj_file_path.strip()
        
        usd_file_path = obj_file_path.replace('.obj', '_obj.usd')
        
        usd_paths.append(usd_file_path)
    
    # 将转换后的路径写入目标 usd 路径文件
    os.makedirs(os.path.dirname(usd_path), exist_ok=True)  # 确保目标目录存在
    with open(usd_path, 'w') as usd_file:
        usd_file.writelines([path + '\n' for path in usd_paths])
    
    print(f"转换完成，USD 路径已写入: {usd_path}")

if __name__ == '__main__':
    obj_path = "unigarment/collect/collect_cd/prepare/ls_tops/ls_tops.txt"
    usd_path = "unigarment/collect/collect_cd/prepare/ls_tops/ls_tops.txt"
    
    transform_obj_to_usd_txt(obj_path, usd_path)

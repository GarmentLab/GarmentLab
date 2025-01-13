import os
import numpy as np
import math
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Slerp, Rotation


def get_unique_filename(base_filename, extension=".png"):
    counter = 0
    filename = f"{base_filename}_{counter}{extension}"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_filename}_{counter}{extension}"

    return filename

def float_truncate(num):
    '''
    Keep four decimal places.
    '''
    result = math.trunc(num * 1e4) / 1e4

    return result


def dense_trajectory_points_generation(start_pos:np.ndarray, end_pos:np.ndarray, start_quat:np.ndarray=None, end_quat:np.ndarray=None, num_points:int=50):
    '''
    generate dense trajectory points for inverse kinematics control.
    '''
    # ---- 1. 生成五个采样点（包括起点和终点） ----
    distance = np.linalg.norm(end_pos - start_pos)
    # print(distance)
    initial_sample_points_num = 5
    initial_sample_points = np.linspace(start_pos, end_pos, initial_sample_points_num)

    # ---- 2. 使用 B 样条拟合采样点，生成平滑轨迹 ----
    tck, u = splprep(initial_sample_points.T, s=0)  # B样条拟合
    u_new = np.linspace(0, 1, num_points)  # 更细致的采样
    interp_pos = np.array(splev(u_new, tck)).T  # 插值后的平滑轨迹
    # print(interp_pos.shape)

    # ---- 3. 对旋转四元数进行球面线性插值 (Slerp) ----
    if start_quat is not None and end_quat is not None:
        rotations = Rotation.from_quat([start_quat, end_quat])  # 四元数转换为旋转对象
        slerp = Slerp([0, 1], rotations)  # Slerp 插值器
        interp_times = np.linspace(0, 1, num_points)  # 插值时间点
        interp_rotations = slerp(interp_times).as_quat()  # 插值结果转换为四元数
        print(interp_rotations.shape)

        return interp_pos, interp_rotations

    return interp_pos

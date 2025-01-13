import os
import sys
sys.path.append(os.getcwd()) # change to your specific path
import numpy as np
import torch
import open3d as o3d
import random
import imageio
import imageio.v3 as iio
import cv2
import time
import omni.replicator.core as rep
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from Env.Utils.code import get_unique_filename, dense_trajectory_points_generation
from Env.Utils.pointcloud import furthest_point_sampling


class Recording_Camera:
    def __init__(self, camera_position:np.ndarray=np.array([0.0, 6.0, 2.6]), camera_orientation:np.ndarray=np.array([0, 20.0, -90.0]), frequency=20, resolution=(640, 480), prim_path="/World/recording_camera"):
        # define camera parameters
        self.camera_position = camera_position
        self.camera_orientation = camera_orientation
        self.frequency = frequency
        self.resolution = resolution
        self.camera_prim_path = prim_path
        # define capture photo flag
        self.capture = True

        # define camera
        self.camera = Camera(
            prim_path=self.camera_prim_path,
            position=self.camera_position,
            orientation=euler_angles_to_quat(self.camera_orientation, degrees=True),
            frequency=self.frequency,
            resolution=self.resolution,
        )

        # Attention: Remember to initialize camera before use in your main code. And Remember to initialize camera after reset the world!!

    def initialize(self, depth_enable:bool=False, pc_enable:bool=False, segment_prim_path_list=None):

        self.video_frame = []
        self.camera.initialize()

        # choose whether add depth attribute or not
        if depth_enable:
            self.camera.add_distance_to_image_plane_to_frame()

        # choose whether add pointcloud attribute or not
        if pc_enable:
            for path in segment_prim_path_list:
                semantic_type = "class"
                semantic_label = path.split("/")[-1]
                print(semantic_label)
                prim_path = path
                print(prim_path)
                rep.modify.semantics([(semantic_type, semantic_label)], prim_path)

            self.render_product = rep.create.render_product(self.camera_prim_path, [640, 480])
            self.annotator = rep.AnnotatorRegistry.get_annotator("pointcloud")
            # self.annotator_semantic = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
            self.annotator.attach(self.render_product)
            # self.annotator_semantic.attach(self.render_product)


    def get_rgb_graph(self, save_or_not:bool=False, save_path:str=get_unique_filename(base_filename=f"./image",extension=".png")):
        '''
        get RGB graph data from recording_camera, save it to be image file(optional).
        Args:
            save_or_not(bool): save or not
            save_path(str): The path you wanna save, remember to add file name and file type(suffix).
        '''
        data = self.camera.get_rgb()
        if save_or_not:
            imageio.imwrite(save_path, data)
        return data

    def get_pointcloud_from_depth(self, show_original_pc_online:bool=False, show_downsample_pc_online:bool=False):
        point_cloud = self.camera.get_pointcloud()
        if show_original_pc_online:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            o3d.visualization.draw_geometries([pcd])
        mask = (point_cloud[:, 2] > 0.005)
        point_cloud = point_cloud[mask]
        down_sampled_point_cloud = furthest_point_sampling(point_cloud, colors=None, n_samples=2048)
        if show_downsample_pc_online:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(down_sampled_point_cloud)
            o3d.visualization.draw_geometries([pcd])
        down_sampled_point_cloud = np.hstack((down_sampled_point_cloud, np.zeros((down_sampled_point_cloud.shape[0], 3))))
        return down_sampled_point_cloud


    def collect_rgb_graph_for_gif(self):
        '''
        take RGB graph from recording_camera and collect them for gif generation.
        '''
        # when capture flag is True, make camera capture photos
        while self.capture:
            data = self.camera.get_rgb()
            if len(data):
                self.video_frame.append(data)

            # take rgb photo every 500 ms
            time.sleep(0.1)
            # print("get rgb successfully")
        print("stop get rgb")


    def create_gif(self, save_path:str=get_unique_filename(base_filename=f"Assets/Replays/carry_garment/animation/animation",extension=".gif")):
        '''
        create gif according to video frame list.
        Args:
            save_path(str): The path you wanna save, remember to include file name and file type(suffix).
        '''
        self.capture = False
        with imageio.get_writer(save_path, mode='I', duration=0.1) as writer:
            for frame in self.video_frame:
                # write each video frame into gif
                writer.append_data(frame)

        print(f"GIF has been save into {save_path}")
        # clear video frame list
        self.video_frame.clear()

    def create_mp4(self, save_path:str=None, fps:int=10):
        '''
        create mp4 according to video frame list. (not mature yet, don't use)
        Args:
            save_path(str): The path you wanna save, remember to include file name and file type(suffix).
        '''
        assert save_path is not None, "save_path can't be None"
        self.capture = False
        height, width, layers = self.video_frame[0].shape
        size = (width, height)
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for frame in self.video_frame:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 转换为 BGR 格式
        out.release()
        print(f"MP4 has been save into {save_path}")
        # clear video frame list
        self.video_frame.clear()

    def get_point_cloud_data(self, save_or_not:bool=True, save_path:str=get_unique_filename(base_filename=f"./pc",extension=".pcd"), real_time_watch:bool=False):
        '''
        get point_cloud's data and color(between[0, 1]) of each point, down_sample the number of points to be 2048, save it to be ply file(optional).
        '''
        self.data=self.annotator.get_data()
        self.point_cloud=np.array(self.data["data"])
        pointRgb=np.array(self.data["info"]['pointRgb'].reshape((-1, 4)))
        self.colors = np.array(pointRgb[:, :3] / 255.0)

        # print(self.point_cloud)
        self.point_cloud, self.colors = furthest_point_sampling(self.point_cloud, self.colors, 2048)


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        if real_time_watch:
            o3d.visualization.draw_geometries([pcd])
        if save_or_not:
            o3d.io.write_point_cloud(save_path, pcd)

        return self.point_cloud, self.colors

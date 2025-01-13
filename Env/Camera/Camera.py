import sys
import numpy as np
import time
from copy import deepcopy
import cv2
import numpy as np
from isaacsim import SimulationApp
import torch
import sys
sys.path.append("/home/luhr/Tactile/IsaacTac/")
import open3d as o3d
from omni.isaac.core import World
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.sensor import Camera
import omni.replicator.core as rep
from omni.isaac.core.utils.rotations import euler_angles_to_quat
# from dgl.geometry import farthest_point_sampler


class MyCamera:
    def __init__(self,world:World,):
        self.world=world
        self.camera_handle_path=find_unique_string_name("/World/Camera",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.camera_handle=self.world.scene.add(XFormPrim(
            prim_path=find_unique_string_name(self.camera_handle_path,is_unique_fn=lambda x: not is_prim_path_valid(x)),
            name=find_unique_string_name("Camera",is_unique_fn=lambda x: not self.world.scene.object_exists(x)),
        ))
        self.camera1_xform_path=find_unique_string_name(self.camera_handle_path+"/Camera1",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.camera_1_path=find_unique_string_name(self.camera1_xform_path+"/Camera",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.camera1_xform=self.world.scene.add(XFormPrim(
            prim_path=self.camera1_xform_path,
            name=find_unique_string_name("Camera1",is_unique_fn=lambda x: not self.world.scene.object_exists(x)),
        ))
        self.camera_1=Camera(
            prim_path=self.camera_1_path,
            resolution=[360,360],
        )
        # self.camera_1.set_focal_length(6)
        # self.camera_1.set_focus_distance(300)
        self.camera_1.set_stereo_role("mono")


    def camera_reset(self):
        self.camera_1.initialize()
        self.camera_1.add_distance_to_image_plane_to_frame()
        self.camera_1.add_semantic_segmentation_to_frame()
        self.camera_1.add_pointcloud_to_frame(False)
        self.camera_1.add_distance_to_camera_to_frame()


        self.camera_1.post_reset()
        self.camera_1.set_local_pose(np.array([0,0,0]),euler_angles_to_quat(np.array([0,0,0])),camera_axes="usd")
        self.camera1_xform.set_world_pose(np.array([0.3,0,0]),orientation=np.array([1,0,0,0]))

        self.render_product_1=rep.create.render_product(self.camera_1_path,[360,360])
        self.annotator_1=rep.AnnotatorRegistry.get_annotator("pointcloud")
        self.annotator_1.attach(self.render_product_1)



    def get_pcd(self,vis:bool=False):
        data1=self.annotator_1.get_data()
        points1=data1["data"].reshape(-1,3)
        colors1=data1["info"]["pointRgb"].reshape(-1,4)/255
        normals1=data1["info"]["pointNormals"].reshape(-1,4)
        semantics1=data1["info"]["pointSemantic"].reshape(-1)
        id1=np.ones_like(semantics1)
        if vis:
            pcd=o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(points1)
            pcd.colors=o3d.utility.Vector3dVector(colors1[:,:3])
            o3d.visualization.draw_geometries([pcd])
        return {"pcd":{"points":points1,"id":id1,"colors":colors1,"semantics":semantics1,"normals":normals1}}

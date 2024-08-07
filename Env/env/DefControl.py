import numpy as np
from omni.isaac.core import World, SimulationContext, PhysicsContext
#simulation_context=SimulationContext(set_defaults=False)
from omni.isaac.core.utils.types import ArticulationAction
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
from omni.isaac.franka import Franka

from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from Env.Utils.transforms import euler_angles_to_quat
from omni.isaac.core.objects import DynamicCuboid,FixedCuboid
from omni.isaac.sensor import Camera
import omni.replicator.core as rep
from Env.Robot.Franka.MyFranka import MyFranka
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim
from pxr import Gf, UsdGeom,Sdf, UsdPhysics, PhysxSchema, UsdLux, UsdShade
from Env.Garment.Garment import Garment
from Env.Deformable.Deformable import Deformable
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
import torch

class AttachmentBlock():
    def __init__(self,world:World, robot:MyFranka, init_place, collision_group=None):
        self.world=world
        self.stage=self.world.stage
        self.name="attach"
        self.init_place=init_place
        self.robot=robot
        self.robot_path=self.robot.get_prim_path()
        self.attachment_path=find_unique_string_name(
            initial_name="/World/Attachment/attach", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self.block_path=self.attachment_path+"/block"
        self.cube_name=find_unique_string_name(initial_name="cube", is_unique_fn=lambda x: not self.world.scene.object_exists(x))
        self.collision_group=collision_group
        self.block_control=self.create()
   


        

    def create(self,):
        
        prim = DynamicCuboid(prim_path=self.block_path, color=np.array([1.0, 0.0, 0.0]),
                    name=self.cube_name,
                    position=self.init_place,
                    scale=np.array([0.02, 0.02, 0.02]),
                    mass=1000,
                    visible=False)
        self.block_prim=prim
        self.world.scene.add(prim)
        self.move_block_controller=self.block_prim._rigid_prim_view
        self.move_block_controller.disable_gravities()
        self.collision_group.CreateIncludesRel().AddTarget(self.block_path)
        return self.move_block_controller


    def set_velocities(self,velocities):
        velocities=velocities.reshape(1,6)
        self.move_block_controller.set_velocities(velocities)
    def set_position(self,grasp_point):

        self.block_prim.set_world_pose(position=grasp_point)

    def get_position(self):
        
        pose,_=self.block_prim.get_world_pose()
        return pose
    
    def attach(self,deformable:Deformable):
        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, self.attachment_path)
        attachment.GetActor0Rel().SetTargets([deformable.deformable_mesh_prim_path])
        attachment.GetActor1Rel().SetTargets([self.block_path])
        att=PhysxSchema.PhysxAutoAttachmentAPI(attachment.GetPrim())
        att.Apply(attachment.GetPrim())
        _=att.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.02)
    
    def detach(self):
        delete_prim(self.attachment_path)

    


class DefControl:
    def __init__(self,world:World,robot:list[MyFranka],deformable:list[Deformable]):
        self.world=world
        self.robot=robot
        self.deformable=deformable
        assert len(self.robot)==len(self.deformable), "The number of robot and deformable must be the same"
        self.stage=self.world.stage
        self.grasp_offset=torch.tensor([0.,0.,-0.01])
        self.collision_group()
        self.attachlist=[None]*len(self.robot)
        
    def collision_group(self):
        self.rigid_group_path="/World/Collision/Rigid_group"
        self.rigid_group = UsdPhysics.CollisionGroup.Define(self.stage, self.rigid_group_path)
        self.filter_rigid=self.rigid_group.CreateFilteredGroupsRel()
        self.robot_group_path="/World/Collision/robot_group"
        self.robot_group = UsdPhysics.CollisionGroup.Define(self.stage, self.robot_group_path)
        self.filter_robot = self.robot_group.CreateFilteredGroupsRel()
        self.deformable_group_path="/World/Collision/deformable_group"
        self.deformable_group = UsdPhysics.CollisionGroup.Define(self.stage, self.deformable_group_path)
        self.filter_deformable = self.deformable_group.CreateFilteredGroupsRel()
        self.attach_group_path="/World/attach_group"
        self.attach_group = UsdPhysics.CollisionGroup.Define(self.stage, self.attach_group_path)
        self.filter_attach = self.attach_group.CreateFilteredGroupsRel()
        self.filter_robot.AddTarget(self.deformable_group_path)
        self.filter_robot.AddTarget(self.rigid_group_path)
        self.filter_robot.AddTarget(self.attach_group_path)
        self.filter_deformable.AddTarget(self.robot_group_path)
        self.filter_deformable.AddTarget(self.rigid_group_path)
        self.filter_deformable.AddTarget(self.attach_group_path)
        self.filter_rigid.AddTarget(self.robot_group_path)
        self.filter_rigid.AddTarget(self.deformable_group_path)
        self.filter_rigid.AddTarget(self.attach_group_path)
        self.filter_attach.AddTarget(self.robot_group_path)
        self.filter_attach.AddTarget(self.deformable_group_path)
        self.filter_attach.AddTarget(self.rigid_group_path)
        self.collectionAPI_robot = Usd.CollectionAPI.Apply(self.filter_robot.GetPrim(), "colliders")
        for robot in self.robot:
            self.collectionAPI_robot.CreateIncludesRel().AddTarget(robot.get_prim_path())
        self.collectionAPI_deformable = Usd.CollectionAPI.Apply(self.filter_deformable.GetPrim(), "colliders")
        self.collectionAPI_deformable.CreateIncludesRel().AddTarget(f"/World/Deformable")
        for deformable in self.deformable:
            self.collectionAPI_deformable.CreateIncludesRel().AddTarget(deformable.deformable_mesh_prim_path)
            self.collectionAPI_deformable.CreateIncludesRel().AddTarget(deformable.deformable_prim_path)
            # self.collectionAPI_deformable.CreateIncludesRel().AddTarget(deformable.particle_system_path)
        self.collectionAPI_attach = Usd.CollectionAPI.Apply(self.filter_attach.GetPrim(), "colliders")
        self.collectionAPI_attach.CreateIncludesRel().AddTarget("/World/Attachment")
        
        self.collectionAPI_rigid = Usd.CollectionAPI.Apply(self.filter_rigid.GetPrim(), "colliders")
       
    
    def make_attachment(self,position:list,flag:list[bool]):
        for i in range(len(self.robot)):
            if flag[i] and self.attachlist[i] is None:
                self.attachlist[i]=AttachmentBlock(self.world,self.robot[i],position[i],collision_group=self.collectionAPI_attach)
            elif flag[i] and self.attachlist[i] is not None:
                continue
            else:
                self.attachlist[i]=None
        print(self.attachlist)

    def robot_goto_position(self,pos:list,ori:list,flag:list[bool],max_limit=300):
        cur_step=0
        self.world.step()
        while 1:
            for i in range(len(self.robot)):
                if flag[i]:
                    self.robot[i].move(pos[i],ori[i])
            self.world.step()
            all_reach_flag=True
            for i in range(len(self.robot)):
                if flag[i]:
                    if not self.robot[i].reach(pos[i],ori[i]):
                        all_reach_flag=False
                        break
            if all_reach_flag or cur_step>max_limit:
                break
            cur_step+=1
    def robot_step(self,pos:list,ori:list,flag:list[bool]):
        for i in range(len(self.robot)):
            if flag[i]:
                self.robot[i].move(pos[i],ori[i])
            self.world.step()
            
    def attach(self,object_list,flag:list[bool]):
        for i in range(len(object_list)):
            if flag[i]:
                self.robot[i].close()
                self.attachlist[i].attach(object_list[i])
                
    def robot_close(self,flag:list[bool]):
        for i in range(len(self.robot)):
            if flag[i]:
                self.robot[i].close()
    
    def robot_open(self,flag:list[bool]):
        for i in range(len(self.robot)):
            if not flag[i]:
                self.robot[i].open()
    
    def robot_reset(self):
        self.robot_open([False]*len(self.robot))
        
                
    
    
    
    def grasp(self,pos:list,ori:list,flag:list[bool]):
        '''
        grasp_function
        pos: list of robots grasp position
        ori: list of robots grasp orientation
        flag: list of bool, grasp or not
        '''
        self.robot_goto_position(pos,ori,flag)
        self.world.pause()
        self.make_attachment(pos,flag)
        self.attach(self.deformable,flag)
        self.world.play()
        for i in range(30):
            self.world.step()
        self.robot_close(flag)
    
                    
        
        
            
    def move(self,pos:list,ori:list,flag:list[bool],max_limit=500):
        '''
        move_function
        pos: list of robots target position
        ori: list of robots target orientation
        flag: list of bool, grasp or not
        '''
        cur_step=0
        self.world.step()
        while 1:
            self.robot_step(pos,ori,flag)
            self.next_pos_list=[]
            for id in range(len(self.robot)):
                if not flag[id]:
                    continue
                robot_pos,robot_ori=self.robot[id].get_cur_ee_pos()
                if isinstance(robot_pos,np.ndarray):
                    robot_pos=torch.from_numpy(robot_pos)
                if isinstance(robot_ori,np.ndarray):
                    robot_ori=torch.from_numpy(robot_ori)
                a=self.Rotation(robot_ori,self.grasp_offset)
                block_handle:AttachmentBlock=self.attachlist[id]
                block_cur_pos=block_handle.get_position()
                block_cur_pos=torch.from_numpy(block_cur_pos)
                block_next_pos=robot_pos+a
                block_velocity=(block_next_pos-block_cur_pos)/(self.world.get_physics_dt()*3)
                # if torch.norm(block_cur_pos-block_next_pos)<0.01:
                #     block_velocity=torch.zeros_like(block_velocity)
                orientation_ped=torch.zeros_like(block_velocity)
                cmd=torch.cat([block_velocity,orientation_ped],dim=-1)
                block_handle.set_velocities(cmd)
                
            # self.block_reach(self.next_pos_list)
            self.world.step()
            self.world.step()
            all_reach_flag=True
            for i in range(len(self.robot)):
                if flag[i]:
                    if not self.robot[i].reach(pos[i],ori[i]):
                        all_reach_flag=False
                        break
            if all_reach_flag or cur_step>max_limit:
                cmd=torch.zeros(6,)
                for id in range(len(self.robot)):
                    if not flag[id]:
                        continue
                    block_handle:AttachmentBlock=self.attachlist[id]
                    block_handle.set_velocities(cmd)
                break
    def ungrasp(self,flag):
        '''
        ungrasp function
        flag: list of bool, grasp or not
        grasp is True
        '''
        for i in range(len(self.attachlist)):
            self.robot[i].open()
            if self.attachlist[i] is not None and not flag[i]:
                self.attachlist[i].detach()
                self.attachlist[i]=None
        # print(self.attachlist)
                
    def Rotation(self,q,vector):
        q0=q[0].item()
        q1=q[1].item()
        q2=q[2].item()
        q3=q[3].item()
        R=torch.tensor(
            [
                [1-2*q2**2-2*q3**2,2*q1*q2-2*q0*q3,2*q1*q3+2*q0*q2],
                [2*q1*q2+2*q0*q3,1-2*q1**2-2*q3**2,2*q2*q3-2*q0*q1],
                [2*q1*q3-2*q0*q2,2*q2*q3+2*q0*q1,1-2*q1**2-2*q2**2],
            ]
        )
        vector=torch.mm(vector.unsqueeze(0),R.transpose(1,0))
        return vector.squeeze(0)
    
    def change_deformable(self,id,deformable):
        self.deformable[id]=deformable
            
            
        
        
        
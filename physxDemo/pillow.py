from isaacsim import SimulationApp
import torch

simulation_app = SimulationApp({"headless": False})

from Env.Config.InflatebleConfig import InflatebleConfig
from Env.Inflatable.Inflatable import Inflatable
import numpy as np

import numpy as np
import torch
from Env.Utils.transforms import quat_diff_rad
from Env.env.BaseEnv import BaseEnv
from omni.isaac.core.objects import DynamicCuboid


if __name__=="__main__":
    env=BaseEnv()
    
    pillow_config1=InflatebleConfig(pos=np.array([-1,0,0.5]),pressure=8)
    pillow1=Inflatable(world=env.world,inflatebleconfig=pillow_config1)

    pillow_config2=InflatebleConfig(pos=np.array([0,0,0.5]),pressure=40)
    pillow2=Inflatable(world=env.world,inflatebleconfig=pillow_config2,particle_system=pillow1.particle_system)

    pillow_config3=InflatebleConfig(pos=np.array([1,0,0.5]),pressure=80)
    pillow3=Inflatable(world=env.world,inflatebleconfig=pillow_config3)

    cube1=env.world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cubes/cube1",
            name="cube1",
            position=np.array([-1,0,1]),
            scale=np.array([0.05,0.05,0.05]),
            color=np.array([0,0,255]),
        )
    )
    
    cube2=env.world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cubes/cube2",
            name="cube2",
            position=np.array([0,0,1]),
            scale=np.array([0.05,0.05,0.05]),
            color=np.array([0,0,255]),
        )
    )

    cube3=env.world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cubes/cube3",
            name="cube3",
            position=np.array([1,0,1]),
            scale=np.array([0.05,0.05,0.05]),
            color=np.array([0,0,255]),

        )
    )

    env.reset()

    while 1:
        env.step()

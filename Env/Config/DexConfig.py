import numpy as np
from Env.env.BaseEnv import BaseEnv
from omni.isaac.kit import SimulationApp

class DexConfig:
    def __init__(self, env: BaseEnv, app: SimulationApp, name = None, prim_path = None, translation: np.ndarray = None, orientation: np.ndarray = None):
        self.env = env
        self.app = app
        self.name = name
        self.prim_path = prim_path

        self.translation = np.array([0.0, 0.0, 0.0]) if translation is None else translation
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation
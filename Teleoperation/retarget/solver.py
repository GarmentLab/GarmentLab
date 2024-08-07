from itertools import combinations
from typing import Optional

import nlopt
import numpy as np
import torch
from shadow_hand import ShadowHandModule


def tsv_tip_palm(keypoints: torch.Tensor) -> torch.Tensor:
    return keypoints[..., [4, 8, 12, 16, 20], :] - keypoints[..., [0], :]


def tsv_middle_palm(keypoints: torch.Tensor) -> torch.Tensor:
    return keypoints[..., [3, 7, 11, 15, 19], :] - keypoints[..., [0], :]


def tsv_tips(keypoints: torch.Tensor) -> torch.Tensor:
    tip_indices = [4, 8, 12, 16, 20]
    tip_pairs = list(combinations(tip_indices, 2))
    start, end = zip(*tip_pairs)
    return keypoints[..., end, :] - keypoints[..., start, :]


def position_tips(keypoints: torch.Tensor) -> torch.Tensor:
    return keypoints[..., [4, 8, 12, 16, 20], :]


def position_middles(keypoints: torch.Tensor) -> torch.Tensor:
    return keypoints[..., [3, 7, 11, 15, 19], :]


def position_middles_and_tips(keypoints: torch.Tensor) -> torch.Tensor:
    return keypoints[..., [3, 4, 7, 8, 11, 12, 15, 16, 19, 20], :]


class MotionMapper(object):
    def __init__(self, side: Optional[str] = "right"):
        self.shadow_hand = ShadowHandModule(side)
        self.loss_fn = torch.nn.SmoothL1Loss(beta=0.01, reduction="none")
        self.optimizer = self.create_optimizer()
        self.latest = np.zeros(self.dof)

    @property
    def dof(self) -> int:
        return self.shadow_hand.dof

    def create_objective_function(self, target: np.ndarray, latest: np.ndarray):
        target: torch.Tensor = torch.from_numpy(target).float()
        latest: torch.Tensor = torch.from_numpy(latest).float()

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            x: torch.Tensor = torch.from_numpy(x.copy()).float()
            x.requires_grad_(True)

            keypoints = self.shadow_hand.forward(x)

            # calculate loss
            loss = self.loss_fn(
                position_middles_and_tips(keypoints), position_middles_and_tips(target)
            )
            # if self.weights is not None:
            #     loss = torch.sum(loss * self.weights) / torch.sum(self.weights)
            # else:
            loss = torch.mean(loss)

            # add regularization
            loss += 1e-3 * torch.sum((x - latest) ** 2)

            # calculate gradient
            loss.backward()
            if grad.size > 0:
                grad[:] = x.grad.numpy()

            return loss.item()

        return objective

    def inequality_constraint(self, finger: str):
        assert finger in ["FF", "MF", "RF", "LF"]
        distal_index = self.shadow_hand.index_of_joint(f"{finger}J1")
        middle_index = self.shadow_hand.index_of_joint(f"{finger}J2")

        def constraint(x: np.ndarray, grad: np.ndarray) -> float:
            # distal joint should be less than middle joint
            if grad.size > 0:
                grad[:] = np.zeros(self.dof)
                grad[distal_index] = 1
                grad[middle_index] = -1
            return x[distal_index] - x[middle_index]

        return constraint

    def create_optimizer(self) -> nlopt.opt:
        optimizer = nlopt.opt(nlopt.LD_SLSQP, self.dof)
        min_rad = self.shadow_hand.min_rad.data.clone().detach().numpy()
        max_rad = self.shadow_hand.max_rad.data.clone().detach().numpy()
        optimizer.set_lower_bounds(min_rad)
        optimizer.set_upper_bounds(max_rad)
        optimizer.add_inequality_constraint(self.inequality_constraint("FF"))
        optimizer.add_inequality_constraint(self.inequality_constraint("MF"))
        optimizer.add_inequality_constraint(self.inequality_constraint("RF"))
        optimizer.add_inequality_constraint(self.inequality_constraint("LF"))
        return optimizer

    def solve(self, target: np.ndarray, latest: Optional[np.ndarray] = None):
        latest = latest if latest is not None else np.zeros(self.dof)
        self.optimizer.set_min_objective(self.create_objective_function(target, latest))
        self.optimizer.set_ftol_abs(1e-5)
        x = self.optimizer.optimize(latest)
        return x

    def get_pose(self, angles: np.ndarray) -> np.ndarray:
        angles = torch.from_numpy(angles).float()
        keypoints = self.shadow_hand.forward(angles)
        return keypoints.detach().numpy()

    def get_zero_pose(self) -> np.ndarray:
        return self.get_pose(np.zeros(self.dof))

    def step(self, target: np.ndarray) -> np.ndarray:
        latest = self.latest.copy()
        objective_function = self.create_objective_function(target, latest)
        self.optimizer.set_min_objective(objective_function)
        self.optimizer.set_ftol_abs(1e-5)
        self.latest = self.optimizer.optimize(latest)
        return self.latest.copy()


if __name__ == "__main__":
    import glob

    from natsort import natsorted

    filenames = glob.glob("hand_pose/*joint*.npy")
    filenames = natsorted(filenames)
    target = np.stack([np.load(filename) for filename in filenames])
    target = target - target[:, 0:1, :]

    # shadow = ShadowHandModule()
    # zero_pose = shadow.zero_pose()
    # print(zero_pose)

    solver = MotionMapper()

    zero_pose = solver.shadow_hand.zero_pose()

    print(zero_pose)

    print(solver.get_zero_pose())

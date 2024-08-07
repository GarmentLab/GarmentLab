from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


@torch.jit.script
def hat(x: torch.Tensor) -> torch.Tensor:
    x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]
    zeros = torch.zeros_like(x0)
    return torch.stack(
        [
            torch.stack([zeros, -x2, x1], dim=-1),
            torch.stack([x2, zeros, -x0], dim=-1),
            torch.stack([-x1, x0, zeros], dim=-1),
        ],
        dim=-2,
    )


@torch.jit.script
def rotation_matrix_from_angle_axis(
    angle: torch.Tensor, axis: torch.Tensor
) -> torch.Tensor:
    assert angle.device == axis.device
    sin, cos = torch.sin(angle), torch.cos(angle)
    sin, cos = sin[..., None, None], cos[..., None, None]
    omega_hat = hat(axis)
    eye = torch.eye(3, device=angle.device, dtype=torch.float32)
    return eye + sin * omega_hat + (1 - cos) * torch.matmul(omega_hat, omega_hat)


@torch.jit.script
def multiply_transform(
    transform1: Tuple[torch.Tensor, torch.Tensor],
    transform2: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    R1, t1 = transform1
    R2, t2 = transform2
    return torch.matmul(R1, R2), torch.mv(R1, t2) + t1


@torch.jit.script
def rotation_matrix_from_quaternion(quat: torch.Tensor) -> torch.Tensor:
    """
    Construct rotation matrix from quaternion
    """
    # Shortcuts for individual elements (using wikipedia's convention)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Set individual elements
    R00 = 1.0 - 2.0 * (y**2 + z**2)
    R01 = 2 * (x * y - z * w)
    R02 = 2 * (x * z + y * w)
    R10 = 2 * (x * y + z * w)
    R11 = 1.0 - 2.0 * (x**2 + z**2)
    R12 = 2 * (y * z - x * w)
    R20 = 2 * (x * z - y * w)
    R21 = 2 * (y * z + x * w)
    R22 = 1.0 - 2.0 * (x**2 + y**2)

    R0 = torch.stack([R00, R01, R02], dim=-1)
    R1 = torch.stack([R10, R11, R12], dim=-1)
    R2 = torch.stack([R20, R21, R22], dim=-1)

    R = torch.stack([R0, R1, R2], dim=-2)

    return R


class Joint:
    name: str
    parent: int
    axis: List[float]
    min_rad: float
    max_rad: float

    def __init__(
        self, name: str, parent: int, axis: List[float], min_rad: float, max_rad: float
    ) -> None:
        self.name = name
        self.parent = parent
        self.axis = axis
        self.min_rad = min_rad
        self.max_rad = max_rad


class Node:
    name: str
    parent: int
    translation: List[float]
    quat: Optional[List[float]] = None

    def __init__(
        self,
        name: str,
        parent: int,
        translation: List[float],
        quat: Optional[List[float]] = None,
    ) -> None:
        self.name = name
        self.parent = parent
        self.translation = translation
        self.quat = quat


def angle_tensor_to_dict(angles: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    assert isinstance(angles, torch.Tensor) or isinstance(angles, np.ndarray)
    if isinstance(angles, torch.Tensor):
        angles = angles.detach().cpu().numpy()
    assert angles.shape == (24,)

    return {
        "rh_WRJ2": angles[0],
        "rh_WRJ1": angles[1],
        "rh_FFJ4": angles[2],
        "rh_FFJ3": angles[3],
        "rh_FFJ2": angles[4],
        "rh_FFJ1": angles[5],
        "rh_MFJ4": angles[6],
        "rh_MFJ3": angles[7],
        "rh_MFJ2": angles[8],
        "rh_MFJ1": angles[9],
        "rh_RFJ4": angles[10],
        "rh_RFJ3": angles[11],
        "rh_RFJ2": angles[12],
        "rh_RFJ1": angles[13],
        "rh_LFJ5": angles[14],
        "rh_LFJ4": angles[15],
        "rh_LFJ3": angles[16],
        "rh_LFJ2": angles[17],
        "rh_LFJ1": angles[18],
        "rh_THJ5": angles[19],
        "rh_THJ4": angles[20],
        "rh_THJ3": angles[21],
        "rh_THJ2": angles[22],
        "rh_THJ1": angles[23],
    }


class ShadowHandModule(nn.Module):
    def __init__(self, side: str = "right") -> None:
        super().__init__()
        assert side in ["right", "left"]

        if side == "right":
            nodes, joints = self.create_right_hand_nodes_and_joints()
        else:
            nodes, joints = self.create_left_hand_nodes_and_joints()

        thbase_quat = torch.Tensor(nodes[23].quat)
        thdistal_quat = torch.Tensor(nodes[27].quat)

        thbase_rotation = rotation_matrix_from_quaternion(thbase_quat)
        thdistal_rotation = rotation_matrix_from_quaternion(thdistal_quat)

        self.thbase_rotation = nn.Parameter(thbase_rotation, requires_grad=False)
        self.thdistal_rotation = nn.Parameter(thdistal_rotation, requires_grad=False)

        self.nodes = nodes
        self.joints = joints

        self.dof: int = len(joints)

        axes = torch.stack([torch.tensor(j.axis) for j in joints], dim=0)
        self.axes = nn.Parameter(axes, requires_grad=False)

        min_rad = torch.tensor([j.min_rad for j in joints])
        max_rad = torch.tensor([j.max_rad for j in joints])
        self.min_rad = nn.Parameter(min_rad, requires_grad=False)
        self.max_rad = nn.Parameter(max_rad, requires_grad=False)

        translation = torch.stack([torch.tensor(n.translation) for n in nodes], dim=0)
        self.translation = nn.Parameter(translation, requires_grad=False)

    @staticmethod
    def create_left_hand_nodes_and_joints() -> Tuple[List[Node], List[Joint]]:
        """Create nodes and joints for left hand."""
        nodes: List[Node] = []
        joints: List[Joint] = []

        nodes.append(Node("wrist", -1, [0.0, 0.0, 0.0]))
        nodes.append(Node("palm", 0, [0.0, 0.0, 0.0340]))
        nodes.append(Node("ffknuckle", 1, [-0.0330, 0.0, 0.0950]))
        nodes.append(Node("ffproximal", 2, [0.0, 0.0, 0.0]))
        nodes.append(Node("ffmiddle", 3, [0.0, 0.0, 0.0450]))
        nodes.append(Node("ffdistal", 4, [0.0, 0.0, 0.0250]))
        nodes.append(Node("fftip", 5, [0.0, 0.0, 0.0260]))
        nodes.append(Node("mfknuckle", 1, [-0.0110, 0.0, 0.0990]))
        nodes.append(Node("mfproximal", 7, [0.0, 0.0, 0.0]))
        nodes.append(Node("mfmiddle", 8, [0.0, 0.0, 0.0450]))
        nodes.append(Node("mfdistal", 9, [0.0, 0.0, 0.0250]))
        nodes.append(Node("mftip", 10, [0.0, 0.0, 0.0260]))
        nodes.append(Node("rfknuckle", 1, [0.0110, 0.0, 0.0950]))
        nodes.append(Node("rfproximal", 12, [0.0, 0.0, 0.0]))
        nodes.append(Node("rfmiddle", 13, [0.0, 0.0, 0.0450]))
        nodes.append(Node("rfdistal", 14, [0.0, 0.0, 0.0250]))
        nodes.append(Node("rftip", 15, [0.0, 0.0, 0.0260]))
        nodes.append(Node("lfmetacarpal", 1, [0.0330, 0.0, 0.02071]))
        nodes.append(Node("lfknuckle", 17, [0.0, 0.0, 0.06579]))
        nodes.append(Node("lfproximal", 18, [0.0, 0.0, 0.0]))
        nodes.append(Node("lfmiddle", 19, [0.0, 0.0, 0.0450]))
        nodes.append(Node("lfdistal", 20, [0.0, 0.0, 0.0250]))
        nodes.append(Node("lftip", 21, [0.0, 0.0, 0.0260]))
        nodes.append(Node("thbase", 1, [-0.0340, -0.0085, 0.0290]))
        nodes.append(Node("thproximal", 23, [0.0, 0.0, 0.0]))
        nodes.append(Node("thhub", 24, [0.0, 0.0, 0.0380]))
        nodes.append(Node("thmiddle", 25, [0.0, 0.0, 0.0]))
        nodes.append(Node("thdistal", 26, [0.0, 0.0, 0.0320]))
        nodes.append(Node("thtip", 27, [0.0, 0.0, 0.0275]))

        nodes[23].quat = [0.0, 0.382683, 0.0, -0.92388]
        nodes[27].quat = [0.707107, 0.0, 0.0, -0.707107]

        joints.append(Joint("WRJ2", 0, [0.0, 1.0, 0.0], -0.523599, 0.174533))
        joints.append(Joint("WRJ1", 1, [1.0, 0.0, 0.0], -0.698132, 0.488692))
        joints.append(Joint("FFJ4", 2, [0.0, 1.0, 0.0], -0.349066, 0.349066))
        joints.append(Joint("FFJ3", 3, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("FFJ2", 4, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("FFJ1", 5, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("MFJ4", 7, [0.0, 1.0, 0.0], -0.349066, 0.349066))
        joints.append(Joint("MFJ3", 8, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("MFJ2", 9, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("MFJ1", 10, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("RFJ4", 12, [0.0, -1.0, 0.0], -0.349066, 0.349066))
        joints.append(Joint("RFJ3", 13, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("RFJ2", 14, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("RFJ1", 15, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("LFJ5", 17, [0.573576, 0.0, -0.819152], 0.0, 0.785398))
        joints.append(Joint("LFJ4", 18, [0.0, -1.0, 0.0], -0.349066, 0.349066))
        joints.append(Joint("LFJ3", 19, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("LFJ2", 20, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("LFJ1", 21, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("THJ5", 23, [0.0, 0.0, 1.0], -1.0472, 1.0472))
        joints.append(Joint("THJ4", 24, [-1.0, 0.0, 0.0], 0.0, 1.22173))
        joints.append(Joint("THJ3", 25, [1.0, 0.0, 0.0], -0.20944, 0.20944))
        joints.append(Joint("THJ2", 26, [0.0, -1.0, 0.0], -0.698132, 0.698132))
        joints.append(Joint("THJ1", 27, [1.0, 0.0, 0.0], 0.0, 1.5708))

        return nodes, joints

    @staticmethod
    def create_right_hand_nodes_and_joints() -> Tuple[List[Node], List[Joint]]:
        """Create nodes and joints for right hand."""
        nodes: List[Node] = []
        joints: List[Joint] = []

        nodes.append(Node("wrist", -1, [0.0, 0.0, 0.0]))
        nodes.append(Node("palm", 0, [0.0, 0.0, 0.0340]))
        nodes.append(Node("ffknuckle", 1, [0.0330, 0.0, 0.0950]))
        nodes.append(Node("ffproximal", 2, [0.0, 0.0, 0.0]))
        nodes.append(Node("ffmiddle", 3, [0.0, 0.0, 0.0450]))
        nodes.append(Node("ffdistal", 4, [0.0, 0.0, 0.0250]))
        nodes.append(Node("fftip", 5, [0.0, 0.0, 0.0260]))
        nodes.append(Node("mfknuckle", 1, [0.0110, 0.0, 0.0990]))
        nodes.append(Node("mfproximal", 7, [0.0, 0.0, 0.0]))
        nodes.append(Node("mfmiddle", 8, [0.0, 0.0, 0.0450]))
        nodes.append(Node("mfdistal", 9, [0.0, 0.0, 0.0250]))
        nodes.append(Node("mftip", 10, [0.0, 0.0, 0.0260]))
        nodes.append(Node("rfknuckle", 1, [-0.0110, 0.0, 0.0950]))
        nodes.append(Node("rfproximal", 12, [0.0, 0.0, 0.0]))
        nodes.append(Node("rfmiddle", 13, [0.0, 0.0, 0.0450]))
        nodes.append(Node("rfdistal", 14, [0.0, 0.0, 0.0250]))
        nodes.append(Node("rftip", 15, [0.0, 0.0, 0.0260]))
        nodes.append(Node("lfmetacarpal", 1, [-0.0330, 0.0, 0.02071]))
        nodes.append(Node("lfknuckle", 17, [0.0, 0.0, 0.06579]))
        nodes.append(Node("lfproximal", 18, [0.0, 0.0, 0.0]))
        nodes.append(Node("lfmiddle", 19, [0.0, 0.0, 0.0450]))
        nodes.append(Node("lfdistal", 20, [0.0, 0.0, 0.0250]))
        nodes.append(Node("lftip", 21, [0.0, 0.0, 0.0260]))
        nodes.append(Node("thbase", 1, [0.0340, -0.0085, 0.0290]))
        nodes.append(Node("thproximal", 23, [0.0, 0.0, 0.0]))
        nodes.append(Node("thhub", 24, [0.0, 0.0, 0.0380]))
        nodes.append(Node("thmiddle", 25, [0.0, 0.0, 0.0]))
        nodes.append(Node("thdistal", 26, [0.0, 0.0, 0.0320]))
        nodes.append(Node("thtip", 27, [0.0, 0.0, 0.0275]))

        nodes[23].quat = [0.92388, 0.0, 0.382683, 0.0]
        nodes[27].quat = [0.707107, 0.0, 0.0, -0.707107]

        joints.append(Joint("WRJ2", 0, [0.0, 1.0, 0.0], -0.523599, 0.174533))
        joints.append(Joint("WRJ1", 1, [1.0, 0.0, 0.0], -0.698132, 0.488692))
        joints.append(Joint("FFJ4", 2, [0.0, -1.0, 0.0], -0.349066, 0.349066))
        joints.append(Joint("FFJ3", 3, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("FFJ2", 4, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("FFJ1", 5, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("MFJ4", 7, [0.0, -1.0, 0.0], -0.349066, 0.349066))
        joints.append(Joint("MFJ3", 8, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("MFJ2", 9, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("MFJ1", 10, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("RFJ4", 12, [0.0, 1.0, 0.0], -0.349066, 0.349066))
        joints.append(Joint("RFJ3", 13, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("RFJ2", 14, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("RFJ1", 15, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("LFJ5", 17, [0.573576, 0.0, 0.819152], 0.0, 0.785398))
        joints.append(Joint("LFJ4", 18, [0.0, 1.0, 0.0], -0.349066, 0.349066))
        joints.append(Joint("LFJ3", 19, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("LFJ2", 20, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("LFJ1", 21, [1.0, 0.0, 0.0], 0.0, 1.5708))
        joints.append(Joint("THJ5", 23, [0.0, 0.0, -1.0], -1.0472, 1.0472))
        joints.append(Joint("THJ4", 24, [1.0, 0.0, 0.0], 0.0, 1.22173))
        joints.append(Joint("THJ3", 25, [1.0, 0.0, 0.0], -0.20944, 0.20944))
        joints.append(Joint("THJ2", 26, [0.0, -1.0, 0.0], -0.698132, 0.698132))
        joints.append(Joint("THJ1", 27, [1.0, 0.0, 0.0], 0.0, 1.5708))

        return nodes, joints

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """Forward kinematics.

        Args:
            angles (torch.Tensor): 24 joint angles.

        Returns:
            torch.Tensor: 3D coordinates of 21 keypoints.
        """
        angles = torch.clamp(angles, self.min_rad, self.max_rad)
        rotation = rotation_matrix_from_angle_axis(angles, self.axes)
        translation = self.translation

        rotation_wrist = rotation[..., 0, :, :]
        translation_wrist = translation[..., 0, :]

        rotation_palm, translation_palm = multiply_transform(
            (rotation_wrist, translation_wrist),
            (rotation[..., 1, :, :], translation[..., 1, :]),
        )

        rotation_ffknuckle, translation_ffknuckle = multiply_transform(
            (rotation_palm, translation_palm),
            (rotation[..., 2, :, :], translation[..., 2, :]),
        )
        rotation_ffproximal, translation_ffproximal = multiply_transform(
            (rotation_ffknuckle, translation_ffknuckle),
            (rotation[..., 3, :, :], translation[..., 3, :]),
        )
        rotation_ffmiddle, translation_ffmiddle = multiply_transform(
            (rotation_ffproximal, translation_ffproximal),
            (rotation[..., 4, :, :], translation[..., 4, :]),
        )
        rotation_ffdistal, translation_ffdistal = multiply_transform(
            (rotation_ffmiddle, translation_ffmiddle),
            (rotation[..., 5, :, :], translation[..., 5, :]),
        )
        translation_fftip = translation_ffdistal + torch.mv(
            rotation_ffdistal, translation[..., 6, :]
        )

        rotation_mfknuckle, translation_mfknuckle = multiply_transform(
            (rotation_palm, translation_palm),
            (rotation[..., 6, :, :], translation[..., 7, :]),
        )
        rotation_mfproximal, translation_mfproximal = multiply_transform(
            (rotation_mfknuckle, translation_mfknuckle),
            (rotation[..., 7, :, :], translation[..., 8, :]),
        )
        rotation_mfmiddle, translation_mfmiddle = multiply_transform(
            (rotation_mfproximal, translation_mfproximal),
            (rotation[..., 8, :, :], translation[..., 9, :]),
        )
        rotation_mfdistal, translation_mfdistal = multiply_transform(
            (rotation_mfmiddle, translation_mfmiddle),
            (rotation[..., 9, :, :], translation[..., 10, :]),
        )
        translation_mftip = translation_mfdistal + torch.mv(
            rotation_mfdistal, translation[..., 11, :]
        )

        rotation_rfknuckle, translation_rfknuckle = multiply_transform(
            (rotation_palm, translation_palm),
            (rotation[..., 10, :, :], translation[..., 12, :]),
        )
        rotation_rfproximal, translation_rfproximal = multiply_transform(
            (rotation_rfknuckle, translation_rfknuckle),
            (rotation[..., 11, :, :], translation[..., 13, :]),
        )
        rotation_rfmiddle, translation_rfmiddle = multiply_transform(
            (rotation_rfproximal, translation_rfproximal),
            (rotation[..., 12, :, :], translation[..., 14, :]),
        )
        rotation_rfdistal, translation_rfdistal = multiply_transform(
            (rotation_rfmiddle, translation_rfmiddle),
            (rotation[..., 13, :, :], translation[..., 15, :]),
        )
        translation_rftip = translation_rfdistal + torch.mv(
            rotation_rfdistal, translation[..., 16, :]
        )

        rotation_lfmetacarpal, translation_lfmetacarpal = multiply_transform(
            (rotation_palm, translation_palm),
            (rotation[..., 14, :, :], translation[..., 17, :]),
        )
        rotation_lfknuckle, translation_lfknuckle = multiply_transform(
            (rotation_lfmetacarpal, translation_lfmetacarpal),
            (rotation[..., 15, :, :], translation[..., 18, :]),
        )
        rotation_lfproximal, translation_lfproximal = multiply_transform(
            (rotation_lfknuckle, translation_lfknuckle),
            (rotation[..., 16, :, :], translation[..., 19, :]),
        )
        rotation_lfmiddle, translation_lfmiddle = multiply_transform(
            (rotation_lfproximal, translation_lfproximal),
            (rotation[..., 17, :, :], translation[..., 20, :]),
        )
        rotation_lfdistal, translation_lfdistal = multiply_transform(
            (rotation_lfmiddle, translation_lfmiddle),
            (rotation[..., 18, :, :], translation[..., 21, :]),
        )
        translation_lftip = translation_lfdistal + torch.mv(
            rotation_lfdistal, translation[..., 22, :]
        )

        rotation_thbase, translation_thbase = multiply_transform(
            (rotation_palm, translation_palm),
            (self.thbase_rotation, translation[..., 23, :]),
        )
        rotation_thbase = torch.matmul(rotation_thbase, rotation[..., 19, :, :])
        rotation_thproximal, translation_thproximal = multiply_transform(
            (rotation_thbase, translation_thbase),
            (rotation[..., 20, :, :], translation[..., 24, :]),
        )
        rotation_thhub, translation_thhub = multiply_transform(
            (rotation_thproximal, translation_thproximal),
            (rotation[..., 21, :, :], translation[..., 25, :]),
        )
        rotation_thmiddle, translation_thmiddle = multiply_transform(
            (rotation_thhub, translation_thhub),
            (rotation[..., 22, :, :], translation[..., 26, :]),
        )
        rotation_thdistal, translation_thdistal = multiply_transform(
            (rotation_thmiddle, translation_thmiddle),
            (self.thdistal_rotation, translation[..., 27, :]),
        )
        rotation_thdistal = torch.matmul(rotation_thdistal, rotation[..., 23, :, :])
        translation_thtip = translation_thdistal + torch.mv(
            rotation_thdistal, translation[..., 28, :]
        )

        return (
            torch.stack(
                [
                    translation_palm,
                    translation_thbase,
                    translation_thmiddle,
                    translation_thdistal,
                    translation_thtip,
                    translation_ffknuckle,
                    translation_ffmiddle,
                    translation_ffdistal,
                    translation_fftip,
                    translation_mfknuckle,
                    translation_mfmiddle,
                    translation_mfdistal,
                    translation_mftip,
                    translation_rfknuckle,
                    translation_rfmiddle,
                    translation_rfdistal,
                    translation_rftip,
                    translation_lfknuckle,
                    translation_lfmiddle,
                    translation_lfdistal,
                    translation_lftip,
                ],
                dim=-2,
            )
            - translation_palm.detach().clone()
        )

    def zero_pose(self) -> torch.Tensor:
        angles = torch.zeros(self.dof)
        return self.forward(angles)

    def index_of_joint(self, name: str) -> int:
        joint_names = [j.name for j in self.joints]
        return joint_names.index(name)


if __name__ == "__main__":
    # for i in range(10):
    #     shadow = ShadowHandModule()
    #     result = shadow.forward(torch.ones(23) * 0.05)
    #     print(i)
    #     print(result)

    shadow = ShadowHandModule()
    print(shadow.zero_pose())

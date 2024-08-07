from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from const import HAND_VISULIZATION_LINKS


def save_animation(animation: FuncAnimation, path: str, fps: int = 10, dpi: int = 200):
    if path.endswith(".gif"):
        # save the animation to a gif file
        animation.save(path, fps=fps, dpi=dpi, writer="pillow")
    elif path.endswith(".mp4"):
        # save the animation to a mp4 file
        animation.save(path, fps=fps, dpi=dpi, writer="ffmpeg")
    else:
        raise ValueError(f"Unsupported file type: {path}")


def plot_hand_keypoints(keypoints: np.ndarray):
    assert keypoints.shape[0] == 21
    lines = HAND_VISULIZATION_LINKS

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    for line in lines:
        ax.plot3D(
            keypoints[line, 0],
            keypoints[line, 1],
            keypoints[line, 2],
            "gray",
        )
    ax.scatter3D(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], "black")
    ax.set_xlim3d(-0.2, 0.2)
    ax.set_ylim3d(-0.2, 0.2)
    ax.set_zlim3d(-0.2, 0.2)
    plt.show()


def plot_hand_motion_keypoints(keypoints: np.ndarray, path: Optional[str] = None):
    assert keypoints.ndim == 3

    lines = HAND_VISULIZATION_LINKS

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    def update(frame):
        ax.clear()

        for line in lines:
            ax.plot3D(
                keypoints[frame, line, 0],
                keypoints[frame, line, 1],
                keypoints[frame, line, 2],
                "gray",
            )
        ax.scatter3D(
            keypoints[frame, :, 0],
            keypoints[frame, :, 1],
            keypoints[frame, :, 2],
            "black",
        )
        ax.set_xlim3d(-0.2, 0.2)
        ax.set_ylim3d(-0.2, 0.2)
        ax.set_zlim3d(-0.2, 0.2)
        ax.set_title(f"Frame {frame:03d}", loc="center")

    anim = FuncAnimation(fig, update, frames=keypoints.shape[0], interval=100)
    plt.show()

    if path is not None:
        save_animation(anim, path)


def plot_two_hands_motion_keypoints(
    keypoints1: np.ndarray, keypoints2: np.ndarray, path: Optional[str] = None
):
    assert keypoints1.ndim == 3

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    def update(frame):
        ax.clear()

        for line in HAND_VISULIZATION_LINKS:
            ax.plot3D(
                keypoints1[frame, line, 0],
                keypoints1[frame, line, 1],
                keypoints1[frame, line, 2],
                "gray",
            )
            ax.plot3D(
                keypoints2[frame, line, 0],
                keypoints2[frame, line, 1],
                keypoints2[frame, line, 2],
                "gray",
            )

        ax.scatter3D(
            keypoints1[frame, :, 0],
            keypoints1[frame, :, 1],
            keypoints1[frame, :, 2],
            "black",
        )
        ax.scatter3D(
            keypoints2[frame, :, 0],
            keypoints2[frame, :, 1],
            keypoints2[frame, :, 2],
            "red",
        )
        ax.set_xlim3d(-0.2, 0.2)
        ax.set_ylim3d(-0.2, 0.2)
        ax.set_zlim3d(-0.2, 0.2)
        ax.set_title(f"Frame {frame:03d}", loc="center")

    anim = FuncAnimation(fig, update, frames=keypoints1.shape[0], interval=100)
    plt.show()

    if path is not None:
        save_animation(anim, path)

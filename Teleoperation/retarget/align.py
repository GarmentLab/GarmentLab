from typing import Sequence

import numpy as np
from numba import jit, njit
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i


def umeyama_alignment(
    x: np.ndarray, y: np.ndarray, estimate_scale: bool = True
) -> np.ndarray:
    """Implements the Umeyama alignment algorithm.
    original paper: https://web.stanford.edu/class/cs273/refs/umeyama.pdf

    Args:
        x (np.ndarray): n x m matrix of n m-dimensional points
        y (np.ndarray): n x m matrix of n m-dimensional points
        estimate_scale (bool, optional): set to True to align also the scale.
            Defaults to True.

    Raises:
        ValueError: x and y must have the same shape

    Returns:
        np.ndarray: (m+1) x (m+1) homogeneous transformation matrix that
            maps x on to y
    """
    assert x.shape == y.shape, "x and y must have the same shape"
    assert x.ndim == 2, "x must be a 2D matrix"
    assert x.shape[0] >= x.shape[1], "x must have at least as many rows as columns"

    n, m = x.shape
    x, y = x.copy(), y.copy()

    centroid_x = np.mean(x, axis=0)
    centroid_y = np.mean(y, axis=0)

    x = np.subtract(x, centroid_x)
    y = np.subtract(y, centroid_y)

    sigma_x = np.mean(np.sum(np.power(x, 2), axis=1), axis=0)

    covariance = np.zeros((m, m))
    np.dot(x.T, y, out=covariance)
    np.multiply(covariance, 1.0 / n, out=covariance)

    U, S, V = np.linalg.svd(covariance)

    assert np.count_nonzero(S > np.finfo(S.dtype).eps) >= m - 1

    M = np.eye(m)
    M[m - 1, m - 1] = np.linalg.det(V.T @ U.T)

    # rotation matrix & scale & translation
    R = V.T @ M @ U.T
    s = np.trace(M @ np.diag(S)) / sigma_x if estimate_scale else 1.0
    t = centroid_y - s * R @ centroid_x

    # homogeneous transformation
    T = np.eye(m + 1)
    T[:m, :m] = s * R
    T[:m, m] = t
    return T


def apply_transformation(x: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """Applies a transformation to a set of points.

    Args:
        x (np.ndarray): n x m matrix of n m-dimensional points
        transformation (np.ndarray): (m+1) x (m+1) homogeneous transformation
            matrix

    Returns:
        np.ndarray: n x m matrix of n m-dimensional points
    """
    assert x.ndim == 2, "x must be a 2D matrix"
    assert transformation.ndim == 2, "transformation must be a 2D matrix"
    _, m = x.shape
    assert transformation.shape == (m + 1, m + 1)
    rotation: np.ndarray = transformation[:m, :m]
    translation: np.ndarray = transformation[:m, m]
    return x @ rotation.T + translation


def compose_transformations(transformations: Sequence[np.ndarray]) -> np.ndarray:
    """Composes a sequence of transformations.

    Args:
        transformations (Sequence[np.ndarray]): sequence of (m+1) x (m+1)
            homogeneous transformation matrices

    Returns:
        np.ndarray: (m+1) x (m+1) homogeneous transformation matrix
    """
    if len(transformations) == 0:
        raise ValueError("transformations must not be empty")
    m = transformations[0].shape[0] - 1
    transformation = np.eye(m + 1)
    for t in transformations:
        transformation = transformation @ t
    return transformation


def inverse_transformation(transformation: np.ndarray) -> np.ndarray:
    """Computes the inverse of a transformation.

    Args:
        transformation (np.ndarray): (m+1) x (m+1) homogeneous transformation
            matrix

    Returns:
        np.ndarray: (m+1) x (m+1) homogeneous transformation matrix
    """
    m = transformation.shape[0] - 1
    rotation = transformation[:m, :m]
    translation = transformation[:m, m]
    inverse_rotation = rotation.T
    inverse_translation = -inverse_rotation @ translation
    inverse_transformation = np.eye(m + 1)
    inverse_transformation[:m, :m] = inverse_rotation
    inverse_transformation[:m, m] = inverse_translation
    return inverse_transformation


def rotation_from_transformation(transformation: np.ndarray) -> np.ndarray:
    """Extracts the rotation matrix from a transformation.

    Args:
        transformation (np.ndarray): (m+1) x (m+1) homogeneous transformation
            matrix

    Returns:
        np.ndarray: m x m rotation matrix
    """
    return transformation[:-1, :-1]


def translation_from_transformation(transformation: np.ndarray) -> np.ndarray:
    """Extracts the translation vector from a transformation.

    Args:
        transformation (np.ndarray): (m+1) x (m+1) homogeneous transformation
            matrix

    Returns:
        np.ndarray: m-dimensional translation vector
    """
    return transformation[:-1, -1]


if __name__ == "__main__":
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]])
    b = np.array([[0, 0, 0], [1, 0, 0], [0, -2, 0]])
    c = np.array([[0, 0, 0], [-1, 0, 0], [0, -2, 0]])

    T = umeyama_alignment(a, b)

    print(T)

    print(b)
    print(apply_transformation(a, T))
    print(np.finfo(T.dtype).eps)

    print(apply_transformation(b, inverse_transformation(T)))

    g_ac = compose_transformations(
        [umeyama_alignment(a, b, False), umeyama_alignment(b, c, False)]
    )
    print(apply_transformation(a, g_ac))

import numpy as np


def toroidal_displacement(pos1, pos2, bounds=(800.0, 600.0)):
    """
    Calculates the shortest displacement vector from pos1 to pos2 on a 2D torus
    using the Minimum Image Convention.

    Supports both single 2D points and 2D arrays of positions.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    delta = np.asarray(pos2, dtype=np.float64) - np.asarray(pos1, dtype=np.float64)
    return (delta + bounds / 2.0) % bounds - bounds / 2.0


def toroidal_distance(pos1, pos2, bounds=(800.0, 600.0)):
    """
    Calculates Euclidean distance under Minimum Image Convention on a 2D torus.
    """
    disp = toroidal_displacement(pos1, pos2, bounds)
    if disp.ndim == 1:
        return float(np.linalg.norm(disp))
    return np.linalg.norm(disp, axis=-1)


def toroidal_center_of_mass(positions, bounds=(800.0, 600.0)):
    """
    Computes the true center of mass of 2D positions on a torus using
    directional statistics (circular mean).
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) == 0:
        return np.array([0.0, 0.0], dtype=np.float64)

    width, height = bounds

    theta_x = 2.0 * np.pi * positions[:, 0] / width
    theta_y = 2.0 * np.pi * positions[:, 1] / height

    mean_cos_x, mean_sin_x = np.mean(np.cos(theta_x)), np.mean(np.sin(theta_x))
    mean_cos_y, mean_sin_y = np.mean(np.cos(theta_y)), np.mean(np.sin(theta_y))

    center_x = (np.arctan2(mean_sin_x, mean_cos_x) % (2.0 * np.pi)) * width / (2.0 * np.pi)
    center_y = (np.arctan2(mean_sin_y, mean_cos_y) % (2.0 * np.pi)) * height / (2.0 * np.pi)

    return np.array([center_x, center_y], dtype=np.float64)

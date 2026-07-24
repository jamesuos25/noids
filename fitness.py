# pyrefly: ignore [missing-import]
import numpy as np
from geometry import toroidal_distance, toroidal_center_of_mass


def compute_frame_fitness(boids, bounds):
    """
    Computes the composite flocking metric for a single frame under toroidal boundary conditions.

    Combines three core flocking components:
    1. Alignment (A): Normalized velocity direction vector sum [0, 1].
    2. Cohesion (C): Proximity of agents to the circular mean center of mass [0, 1].
    3. Separation (S): Vectorized multi-neighbor penalty for overcrowding/collisions under Minimum Image Convention [0, 1].

    Returns:
        Composite frame score = Alignment * Cohesion * Separation
    """
    num_boids = len(boids)
    if num_boids == 0:
        return 0.0, 0.0, 0.0, 0.0

    positions = np.array([b.position for b in boids])
    velocities = np.array([b.velocity for b in boids])

    # 1. Alignment Metric (Polarization)
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    speeds[speeds == 0] = 1.0
    normalized_velocities = velocities / speeds
    avg_direction = np.mean(normalized_velocities, axis=0)
    alignment = np.linalg.norm(avg_direction)  # Range [0, 1]

    # 2. Cohesion Metric (Toroidal)
    center_of_mass = toroidal_center_of_mass(positions, bounds)
    distances_to_center = toroidal_distance(center_of_mass, positions, bounds)
    mean_dist_to_center = np.mean(distances_to_center)

    max_expected_radius = bounds[0] * 0.25  # Reference normalization distance (200px)
    cohesion = max(0.0, 1.0 - (mean_dist_to_center / max_expected_radius))

    # 3. Separation Metric (Vectorized Toroidal Multi-Neighbor Separation)
    # Computes pairwise N x N toroidal displacement matrix
    bounds_arr = np.asarray(bounds, dtype=np.float64)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    diff = (diff + bounds_arr / 2.0) % bounds_arr - bounds_arr / 2.0
    dist_matrix = np.linalg.norm(diff, axis=-1)

    # Fill diagonal with infinity so self-interactions are ignored
    np.fill_diagonal(dist_matrix, np.inf)

    # Collision threshold = 2x physical body radius (12px)
    collision_threshold = boids[0].radius * 2.0

    # Linear penalty between 0.0 (overlapping) and 1.0 (safe distance) for all neighbor interactions
    pairwise_scores = np.minimum(1.0, dist_matrix / collision_threshold)

    # Average score across all distinct off-diagonal neighbor pairs
    mask = ~np.eye(num_boids, dtype=bool)
    separation = float(np.mean(pairwise_scores[mask]))

    # Composite Multiplicative Fitness
    frame_fitness = float(alignment * cohesion * separation)
    return frame_fitness, float(alignment), float(cohesion), float(separation)
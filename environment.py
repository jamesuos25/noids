# pyrefly: ignore [missing-import]
import numpy as np
from geometry import toroidal_distance, toroidal_center_of_mass


class Environment:
    """
    2D Simulation Environment.

    Handles:
    - Spatial arena dimensions and toroidal wrapping.
    - Neighbor searching within a perception radius (150px) using Minimum Image Convention.
    - Global and local flock metric calculations (Circular Center of Mass, Average Heading).
    - Non-uniform, turbulent spatial wind vector fields.
    - Wind force magnitude is scaled relative to max_boid_force,
      so that wind_strength=1.0 produces a peak force equal to
      the boid's maximum steering force.
    """
    def __init__(self, width=1200, height=900, wind_strength=0.0, perception_radius=200.0, max_boid_force=0.3, rng=None):
        self.width = width
        self.height = height
        self.bounds = (width, height)
        self.wind_strength = wind_strength
        self.perception_radius = perception_radius
        self.max_boid_force = max_boid_force
        self.frame_count = 0

        # Isolated Wind RNG stream
        self.rng = rng if rng is not None else np.random.default_rng()

        # Wind dynamics state
        self.current_wind_angle = self.rng.uniform(0, 2 * np.pi)
        self.current_wind = np.array([0.0, 0.0], dtype=np.float64)

    def update_wind(self):
        """
        Updates the global atmospheric wind vector.
        Wind direction and strength drift smoothly over time (stochastic turbulence).
        Spatially uniform across the whole arena -> 100% toroidal boundary compatible.
        Wind magnitude is scaled by max_boid_force so that wind_strength=1.0
        produces a peak force equal to the boid's maximum steering capacity.
        Uses isolated wind RNG stream to guarantee exact CRN reproducibility across candidates.
        """
        if self.wind_strength <= 0.0:
            self.current_wind = np.array([0.0, 0.0], dtype=np.float64)
            return

        # 1. Angle drifts smoothly via small random walk (-0.05 to +0.05 radians per frame)
        angle_drift = self.rng.uniform(-0.05, 0.05)
        self.current_wind_angle += angle_drift

        # 2. Magnitude fluctuates around base wind_strength (+/- 20% gust noise)
        #    Scaled by max_boid_force so wind_strength=1.0 matches peak steering force
        gust_factor = 1.0 + self.rng.uniform(-0.2, 0.2)
        magnitude = self.wind_strength * self.max_boid_force * gust_factor

        # 3. Compute 2D wind vector
        self.current_wind = np.array([
            np.cos(self.current_wind_angle) * magnitude,
            np.sin(self.current_wind_angle) * magnitude
        ], dtype=np.float64)

    def get_wind_at_position(self, position):
        """Returns the current global wind vector (same everywhere at frame t)."""
        return self.current_wind

    def compute_flock_metrics(self, boids):
        """
        Calculates group metrics required for boid sensory inputs:
        1. Global Flock Center of Mass (circular mean on torus).
        2. Global Average Heading Angle.
        """
        positions = np.array([b.position for b in boids])
        velocities = np.array([b.velocity for b in boids])

        # 1. Toroidal Center of Mass via circular mean
        flock_center = toroidal_center_of_mass(positions, self.bounds)

        # 2. Average Heading Vector -> Angle
        avg_velocity = np.mean(velocities, axis=0)
        global_avg_heading = np.arctan2(avg_velocity[1], avg_velocity[0])

        return flock_center, global_avg_heading

    def step(self, boids):
        """
        Advances the physical simulation forward by 1 frame (1 step).

        Uses a single vectorized NxN pairwise distance computation to replace
        O(n^2) Python-level neighbor search loops. All neighbor lookups, nearest-
        neighbor finding, and local heading calculations use matrix indexing.

        Process:
        1. Update global dynamic wind (temporal stochastic drift and gusts).
        2. Compute NxN toroidal displacement and distance matrices (single vectorized op).
        3. Compute global flock center of mass and average heading.
        4. For each boid: find local neighbors and nearest via matrix indexing.
        5. Query boid's neural network for internal steering force.
        6. Integrate unified physics update step for all boids simultaneously.
        """
        self.frame_count += 1

        # 1. Update temporal global dynamic wind vector
        self.update_wind()

        num_boids = len(boids)
        positions = np.array([b.position for b in boids])
        velocities = np.array([b.velocity for b in boids])

        # 2. Single vectorized NxN pairwise computation (replaces ~1700 individual calls)
        # disp_matrix[i,j] = shortest displacement vector FROM boid i TO boid j (MIC)
        bounds_arr = np.asarray(self.bounds, dtype=np.float64)
        raw_diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        disp_matrix = (raw_diff + bounds_arr / 2.0) % bounds_arr - bounds_arr / 2.0
        dist_matrix = np.linalg.norm(disp_matrix, axis=-1)
        np.fill_diagonal(dist_matrix, np.inf)

        # 3. Flock-level metrics (inlined to avoid redundant position extraction)
        flock_center = toroidal_center_of_mass(positions, self.bounds)
        avg_velocity = np.mean(velocities, axis=0)
        global_avg_heading = np.arctan2(avg_velocity[1], avg_velocity[0])

        # Store forces computed in this frame before applying them
        steering_forces = []

        for i, boid in enumerate(boids):
            # 4. Neighbor lookup via matrix indexing (replaces O(n) Python loop)
            neighbor_mask = dist_matrix[i] <= self.perception_radius
            neighbor_indices = np.where(neighbor_mask)[0]
            has_neighbors = len(neighbor_indices) > 0

            # Compute local average heading (fall back to global if no local neighbors)
            if has_neighbors:
                avg_v = np.mean(velocities[neighbor_indices], axis=0)
                local_heading = np.arctan2(avg_v[1], avg_v[0])

                # Nearest perceived neighbor via precomputed matrix
                local_nearest = np.argmin(dist_matrix[i, neighbor_indices])
                nearest_global = neighbor_indices[local_nearest]
                nearest_dist = dist_matrix[i, nearest_global]
                nearest_disp = disp_matrix[i, nearest_global]
            else:
                local_heading = global_avg_heading
                nearest_dist = 0.0
                nearest_disp = np.zeros(2)

            # 5. Calculate Neural Steering Force (with precomputed neighbor data)
            steer = boid.compute_steering(
                nearest_dist, nearest_disp, has_neighbors,
                flock_center, local_heading, bounds=self.bounds
            )
            steering_forces.append(steer)

        # 6. Execute unified physics update for all boids simultaneously
        for i, boid in enumerate(boids):
            boid.update(steering_forces[i], self.current_wind, self.bounds)
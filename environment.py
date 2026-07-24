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
    def __init__(self, width=800, height=600, wind_strength=0.0, perception_radius=150.0, max_boid_force=0.3, rng=None):
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

    def get_local_neighbors(self, boid, all_boids):
        """
        Returns a list of all other boids sitting within this boid's 
        perception radius (150 pixels) under minimum image convention.
        """
        neighbors = []
        for choice in all_boids:
            if choice is not boid:
                # Toroidal Euclidean distance calculation
                dist = toroidal_distance(boid.position, choice.position, self.bounds)
                if dist <= self.perception_radius:
                    neighbors.append(choice)
        return neighbors

    def step(self, boids):
        """
        Advances the physical simulation forward by 1 frame (1 step).

        Process:
        1. Update global dynamic wind (temporal stochastic drift and gusts).
        2. Compute global flock center of mass and average heading.
        3. For each boid: gather local neighbors within perception radius.
        4. Calculate local average heading of those neighbors.
        5. Query boid's neural network for internal steering force.
        6. Retrieve current atmospheric wind force at boid's position.
        7. Integrate unified physics update step for all boids simultaneously.
        """
        self.frame_count += 1

        # 1. Update temporal global dynamic wind vector
        self.update_wind()

        # 2. Compute global flock metrics
        flock_center, global_avg_heading = self.compute_flock_metrics(boids)

        # Store forces computed in this frame before applying them
        steering_forces = []
        wind_forces = []

        for boid in boids:
            # Gather local neighbors within perception radius (150px)
            neighbors = self.get_local_neighbors(boid, boids)

            # Compute local average heading (fall back to global if no local neighbors)
            if len(neighbors) > 0:
                local_velocities = np.array([n.velocity for n in neighbors])
                avg_v = np.mean(local_velocities, axis=0)
                local_heading = np.arctan2(avg_v[1], avg_v[0])
            else:
                local_heading = global_avg_heading

            # 1. Calculate Neural Steering Force
            steer = boid.compute_steering(neighbors, flock_center, local_heading, bounds=self.bounds)

            # 2. Retrieve current atmospheric wind force
            wind = self.get_wind_at_position(boid.position)

            steering_forces.append(steer)
            wind_forces.append(wind)

        # Execute unified physics update for all boids simultaneously
        for i, boid in enumerate(boids):
            boid.update(steering_forces[i], wind_forces[i], self.bounds)
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy.stats import truncnorm
from geometry import toroidal_displacement, toroidal_distance


def sample_boid_params(mean_params, sigma, rng=None, lower_bound_std=-3.0, upper_bound_std=3.0):
    """
    Samples individual parameters for a single boid using a Truncated Normal Distribution clamped at +/- 3 standard deviations.
    Uses explicit Generator instance to avoid consuming global RNG state.
    """
    mean_params = np.asarray(mean_params, dtype=np.float64)
    if sigma <= 0.0:
        return mean_params.copy()

    # Truncated normal bounds in standard deviation units
    a, b = lower_bound_std, upper_bound_std
    if rng is None:
        rng = np.random.default_rng()
    return truncnorm.rvs(a, b, loc=mean_params, scale=sigma, random_state=rng)


class BoidNeuralNetwork:
    """
    Feedforward Neural Network Controller (5 Inputs -> 5 Hidden -> 2 Outputs).

    Architecture (42 total parameters):
    - Layer 1 Weights (W1): 5 inputs * 5 hidden = 25 params
    - Layer 1 Biases (B1): 5 hidden = 5 params
    - Layer 2 Weights (W2): 5 hidden * 2 outputs = 10 params
    - Layer 2 Biases (B2): 2 outputs = 2 params
    """
    def __init__(self, params):
        params = np.asarray(params, dtype=np.float64)

        # Explicit matrix slicing
        self.W1 = params[0:25].reshape(5, 5)   # 25 Layer 1 weights
        self.B1 = params[25:30]                # 5 Layer 1 biases
        self.W2 = params[30:40].reshape(5, 2)  # 10 Layer 2 weights
        self.B2 = params[40:42]                # 2 Layer 2 biases

    def forward(self, inputs):
        """
        Passes 5 sensory inputs through the network.
        Returns: torque [-1, 1], thrust [-1, 1]
        """
        inputs = np.asarray(inputs, dtype=np.float64)

        # Layer 1: Input -> Hidden (tanh activation)
        hidden = np.tanh(np.dot(inputs, self.W1) + self.B1)

        # Layer 2: Hidden -> Output (tanh activation)
        outputs = np.tanh(np.dot(hidden, self.W2) + self.B2)

        return outputs[0], outputs[1]


class Boid:
    """
    Physical agent body. Reads relative sensory inputs, queries its neural brain, and updates position using unified physics + environmental wind forces.
    """
    def __init__(self, x, y, mean_params, sigma=0.0, rng=None, spawn_rng=None):
        # Physical State
        self.position = np.array([x, y], dtype=np.float64)

        if spawn_rng is None:
            spawn_rng = np.random.default_rng()

        # Velocity and Speed limits
        angle = spawn_rng.uniform(0, 2 * np.pi)
        self.max_speed = 4.0
        self.min_speed = 1.0
        self.speed = spawn_rng.uniform(self.min_speed, self.max_speed)
        self.velocity = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64) * self.speed

        self.max_force = 0.3
        self.radius = 6.0  # Physical collision boundary

        # Sample unique parameters for this boid using Gene #43 (sigma)
        self.individual_params = sample_boid_params(mean_params, sigma, rng=rng)
        self.brain = BoidNeuralNetwork(self.individual_params)

    def get_heading_angle(self):
        """Returns orientation angle in radians."""
        return np.arctan2(self.velocity[1], self.velocity[0])

    def get_sensory_inputs(self, nearest_dist, nearest_disp, has_neighbors, flock_center, average_heading, max_sensor_dist=200.0, bounds=(800.0, 600.0)):
        """
        Calculates 5 relative polar inputs for the neural network.
        Accepts precomputed nearest-neighbor data from the environment's distance matrix.
        All angles are relative to the boid's current heading [-1, 1].
        """
        heading = self.get_heading_angle()

        # 1 & 2: Nearest Neighbour distance and relative angle (precomputed)
        if has_neighbors:
            norm_nearest_dist = min(nearest_dist / max_sensor_dist, 1.0)

            rel_angle = np.arctan2(nearest_disp[1], nearest_disp[0]) - heading
            rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
            nearest_angle = rel_angle / np.pi  # Normalize to [-1, 1]
        else:
            norm_nearest_dist = 1.0
            nearest_angle = 0.0

        # 3 & 4: Center of Mass distance and relative angle under toroidal boundary
        center_vec = toroidal_displacement(self.position, flock_center, bounds)
        center_dist = min(np.linalg.norm(center_vec) / max_sensor_dist, 1.0)
        center_angle_raw = np.arctan2(center_vec[1], center_vec[0]) - heading
        center_angle = ((center_angle_raw + np.pi) % (2 * np.pi) - np.pi) / np.pi

        # 5: Relative average heading of local group
        rel_avg_heading = ((average_heading - heading + np.pi) % (2 * np.pi) - np.pi) / np.pi

        return np.array([norm_nearest_dist, nearest_angle, center_dist, center_angle, rel_avg_heading])

    def compute_steering(self, nearest_dist, nearest_disp, has_neighbors, flock_center, average_heading, bounds=(800.0, 600.0)):
        """
        Queries the neural controller to generate a 2D steering vector.
        Accepts precomputed nearest-neighbor data from the environment's distance matrix.
        """
        inputs = self.get_sensory_inputs(nearest_dist, nearest_disp, has_neighbors, flock_center, average_heading, bounds=bounds)
        torque, thrust = self.brain.forward(inputs)

        # Convert scalar torque to heading change (+/- 45 degree turn)
        current_heading = self.get_heading_angle()
        steer_angle = current_heading + (torque * (np.pi / 4))

        # Desired velocity calculation
        desired_direction = np.array([np.cos(steer_angle), np.sin(steer_angle)])
        desired_speed = np.clip(self.speed + thrust, self.min_speed, self.max_speed)
        desired_velocity = desired_direction * desired_speed

        # Steering force = desired_velocity - current_velocity
        steering_force = desired_velocity - self.velocity

        # Clamp steering force to max_force
        force_norm = np.linalg.norm(steering_force)
        if force_norm > self.max_force:
            steering_force = (steering_force / force_norm) * self.max_force

        return steering_force

    def update(self, steering_force, wind_force, bounds):
        """
        Unified Physics Integration Step:
        F_total = F_steering + F_wind
        Velocity = Velocity + F_total
        Position = Position + Velocity
        """
        # Sum internal steering and external wind forces
        total_force = steering_force + wind_force
        self.velocity += total_force

        # Clamp speed magnitude within [min_speed, max_speed]
        speed = np.linalg.norm(self.velocity)
        if speed > 0:
            speed_clamped = np.clip(speed, self.min_speed, self.max_speed)
            self.velocity = (self.velocity / speed) * speed_clamped
            self.speed = speed_clamped

        # Update position
        self.position += self.velocity

        # Toroidal boundary wraparound
        width, height = bounds
        self.position[0] %= width
        self.position[1] %= height
import numpy as np
import matplotlib.pyplot as plt

class RK4_sim:
    def __init__(self, particle, field, dt, num_steps):
        self.particle = particle
        self.field = field
        self.dt = dt
        self.num_steps = num_steps

    def lorentz_force(self, velocity, position):
        """Calculate the Lorentz force F = q(v x B) for the particle"""
        B = self.field.parabolic_field(position[0], position[1], position[2])
        return np.cross(velocity, B)

    def rk4_step(self):
        """Perform a single RK4 step to update the particle's position and velocity"""
        q, m = self.particle.charge, self.particle.mass
        v, r = self.particle.velocity, self.particle.position

        k1_v = self.dt * (q / m) * self.lorentz_force(v, r)
        k1_r = self.dt * v

        k2_v = self.dt * (q / m) * self.lorentz_force(v + 0.5 * k1_v, r + 0.5 * k1_r)
        k2_r = self.dt * (v + 0.5 * k1_v)

        k3_v = self.dt * (q / m) * self.lorentz_force(v + 0.5 * k2_v, r + 0.5 * k2_r)
        k3_r = self.dt * (v + 0.5 * k2_v)

        k4_v = self.dt * (q / m) * self.lorentz_force(v + k3_v, r + k3_r)
        k4_r = self.dt * (v + k3_v)

        v_next = v + (1 / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        r_next = r + (1 / 6) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)

        self.particle.update_velocity(v_next)
        self.particle.update_position(r_next)

    def run(self):
        """Run the simulation for the specified number of steps"""
        for _ in range(self.num_steps):
            self.rk4_step()

    def plot_trajectory_with_lorentz_force(self):
        """Plot the particle's 3D trajectory with arrows for Lorentz force"""
        positions = np.array(self.particle.positions)
        velocities = np.array(self.particle.velocities)
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        # Calculate Lorentz forces at each point
        lorentz_forces = -np.array([self.lorentz_force(velo, pos) for velo, pos in zip(velocities,positions)])
        force_magnitudes = np.linalg.norm(lorentz_forces, axis=1)

        field_direction = np.array([self.field.parabolic_field(pos[0], pos[1], pos[2]) for pos in positions])

        # Create a 3D plot for the trajectory
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the particle trajectory
        ax.plot(x, y, z, color='black', label="Particle trajectory", lw=2)

        # Plot arrows for the Lorentz force at every nth point
        n = 50  # Plot an arrow for every nth point to avoid clutter

        ax.quiver(x[::n], y[::n], z[::n],
                  lorentz_forces[::n, 0], lorentz_forces[::n, 1], lorentz_forces[::n, 2],
                  length=0.00000005, normalize=False, color='red', label='Lorentz force')

        ax.quiver(x[::n], y[::n], z[::n],
                  velocities[::n, 0], velocities[::n, 1], velocities[::n, 2],
                  length=0.0005, normalize=True, color='green', label='Velocity')

        ax.quiver(x[::n], y[::n], z[::n],
                  field_direction[::n, 0], field_direction[::n, 1], field_direction[::n, 2],
                  length=0.0005, normalize=True, color='blue', label='Field direction')

        # Set labels and show plot
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_zlabel('Z position (m)')
        ax.set_title('Particle Trajectory with Lorentz Force (3D)')
        ax.legend()

        plt.show()





import numpy as np
import matplotlib.pyplot as plt
from magnetic_field import MagneticField, Quadratic_Field, LHCb_Field

class RK4_sim:
    def __init__(self, particle, field : MagneticField, dt, num_steps):
        self.particle = particle
        self.field = field
        self.dt = dt
        self.num_steps = num_steps

    def lorentz_force(self, momentum, position):
        """Calculate the Lorentz force F = q(v x B) where v = p/m"""
        B = self.field.magnetic_field(position[0], position[1], position[2])
        velocity = momentum / self.particle.mass  # v = p/m
        return self.particle.charge * np.cross(velocity, B)

    def rk4_step(self):
        """Perform a single RK4 step to update the particle's position and momentum"""
        q, m = self.particle.charge, self.particle.mass
        p, r = self.particle.momentum, self.particle.position

        k1_p = self.dt * self.lorentz_force(p, r)
        k1_r = self.dt * (p / m)

        k2_p = self.dt * self.lorentz_force(p + 0.5 * k1_p, r + 0.5 * k1_r)
        k2_r = self.dt * ((p + 0.5 * k1_p) / m)

        k3_p = self.dt * self.lorentz_force(p + 0.5 * k2_p, r + 0.5 * k2_r)
        k3_r = self.dt * ((p + 0.5 * k2_p) / m)

        k4_p = self.dt * self.lorentz_force(p + k3_p, r + k3_r)
        k4_r = self.dt * ((p + k3_p) / m)

        # Update momentum and position
        p_next = p + (1 / 6) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        r_next = r + (1 / 6) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)

        self.particle.update_momentum(p_next)
        self.particle.update_position(r_next)

        print(f'position : {r} velcocity in units of c  : {np.linalg.norm(p / m)/1e6}, B field strength : {self.field.field_strength(self.field.magnetic_field(r[0], r[1], r[2]))}')  # Print the position and velocity

    def run(self):
        """Run the simulation for the specified number of steps"""
        for _ in range(self.num_steps):
            self.rk4_step()

    def plot_trajectory_with_lorentz_force(self):
        """Plot the particle's 3D trajectory with arrows for Lorentz force"""
        positions = np.array(self.particle.positions)
        momenta = np.array(self.particle.momenta)
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        # Calculate Lorentz forces at each point
        lorentz_forces = -np.array([self.lorentz_force(mom, pos) for mom, pos in zip(momenta,positions)])
        force_magnitudes = np.linalg.norm(lorentz_forces, axis=1)

        field_direction = np.array([self.field.magnetic_field(pos[0], pos[1], pos[2]) for pos in positions])

        # Create a 3D plot for the trajectory
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the particle trajectory
        ax.plot(x, y, z, color='black', label="Particle trajectory", lw=2)

        # Plot arrows for the Lorentz force at every nth point
        n = self.num_steps//5  # Plot an arrow for every nth point to avoid clutter

        # ax.quiver(x[::n], y[::n], z[::n],
        #           lorentz_forces[::n, 0], lorentz_forces[::n, 1], lorentz_forces[::n, 2],
        #           length=0.00000005, normalize=True, color='red', label='Lorentz force')

        # ax.quiver(x[::n], y[::n], z[::n],
        #           momenta[::n, 0], momenta[::n, 1], momenta[::n, 2],
        #           length=0.0005, normalize=True, color='green', label='Velocity')

        # ax.quiver(x[::n], y[::n], z[::n],
        #           field_direction[::n, 0], field_direction[::n, 1], field_direction[::n, 2],
        #           length=0.0005, normalize=True, color='blue', label='Field direction')

        # Set labels and show plot
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_zlabel('Z position (m)')
        ax.set_title('Particle Trajectory with Lorentz Force (3D)')
        ax.legend()

        plt.show()





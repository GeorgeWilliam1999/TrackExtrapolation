import numpy as np
import matplotlib.pyplot as plt
from Classes.magnetic_field import MagneticField, Quadratic_Field, LHCb_Field

class RK4_sim:
    def __init__(self, particles, field: MagneticField, dt, num_steps):
        self.particles = particles  # List of particles
        self.field = field
        self.dt = dt
        self.num_steps = num_steps

    def lorentz_force(self,particle, momentum, position):
        """Calculate the Lorentz force F = q(v x B) where v = p/m"""
        B = self.field.interpolated_field(position[0], position[1], position[2])
        velocity = momentum / particle.mass  # v = p/m

        return particle.charge * np.cross(velocity, B)

    def rk4_step(self):
        """Perform a single RK4 step to update the particles' positions and momenta"""
        for particle in self.particles:
            q, m = particle.charge, particle.mass
            p, r = particle.momentum, particle.position

            k1_p = self.dt * self.lorentz_force(particle, p, r)
            k1_r = self.dt * (p / m)

            k2_p = self.dt * self.lorentz_force(particle, p + 0.5 * k1_p, r + 0.5 * k1_r)
            k2_r = self.dt * ((p + 0.5 * k1_p) / m)

            k3_p = self.dt * self.lorentz_force(particle, p + 0.5 * k2_p, r + 0.5 * k2_r)
            k3_r = self.dt * ((p + 0.5 * k2_p) / m)

            k4_p = self.dt * self.lorentz_force(particle, p + k3_p, r + k3_r)
            k4_r = self.dt * ((p + k3_p) / m)

            # Update momentum and position
            p_next = p + (1 / 6) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
            r_next = r + (1 / 6) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)

            particle.update_momentum(p_next)
            particle.update_position(r_next)

            print(f'{particle.Ptype}|  position : {r} velocity in units of c : {np.linalg.norm(p / m)}, B field strength : {self.field.field_strength(self.field.interpolated_field(r[0], r[1], r[2]))}, Lorentz Force : {self.lorentz_force(particle, p, r)}')  # Print the position and velocity

    def run(self):
        """Run the simulation for the specified number of steps"""
        for _ in range(self.num_steps):
            self.rk4_step()

    def plot_trajectory_with_lorentz_force(self):
        """Plot the particles' 3D trajectories with arrows for Lorentz force"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for particle in self.particles:
            positions = np.array(particle.positions)
            momenta = np.array(particle.momenta)
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]

            # Calculate Lorentz forces at each point
            lorentz_forces = np.array([self.lorentz_force(particle, mom, pos) for mom, pos in zip(momenta, positions)])
            force_magnitudes = np.linalg.norm(lorentz_forces, axis=1)

            field_direction = np.array([self.field.interpolated_field(pos[0], pos[1], pos[2]) for pos in positions])

            # Calculate velocities from momenta
            velocities = momenta / particle.mass

            # Plot the particle trajectory
            ax.plot(x, y, z, label=f"{particle.Ptype} trajectory", lw=2)

            # Plot arrows for the Lorentz force at every nth point
            n = self.num_steps // 50  # Plot an arrow for every nth point to avoid clutter

            ax.quiver(x[::n], y[::n], z[::n],
                    lorentz_forces[::n, 0], lorentz_forces[::n, 1], lorentz_forces[::n, 2],
                    length=.1, normalize=True, color='red')

            ax.quiver(x[::n], y[::n], z[::n],
                    velocities[::n, 0], velocities[::n, 1], velocities[::n, 2],
                    length=1, normalize=True, color='green')

            ax.quiver(x[::n], y[::n], z[::n],
                    field_direction[::n, 0], field_direction[::n, 1], field_direction[::n, 2],
                    length=.1, normalize=True, color='blue')

        # Add single legend entries for the velocity and Lorentz force arrows
        ax.quiver([], [], [], [], [], [], color='red', label='Lorentz Force')
        ax.quiver([], [], [], [], [], [], color='green', label='Velocity')
        ax.quiver([], [], [], [], [], [], color='blue', label='Field Direction')

        # Set labels and show plot
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_zlabel('Z position (m)')
        ax.set_title('Particles Trajectories with Lorentz Force (3D)')
        ax.legend()

        plt.show()


class peer_method():
    def __init__():
        pass


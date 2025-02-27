import numpy as np
import matplotlib.pyplot as plt
from Classes.magnetic_field import MagneticField, Quadratic_Field, LHCb_Field
from matplotlib.widgets import CheckButtons
import os 
import datetime

class RK4_sim_dz():
    def __init__(self, particle_states, field: MagneticField, dz, z, num_steps):
        self.particles = particle_states  # List of particles, using the particle_state object
        self.field = field
        self.dz = dz
        self.z = z
        self.num_steps = num_steps
        self.output_dir = "Recorded_trajectories/run_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_state_derivative(self, state, B):
        """Compute the derivative of the state vector using the Lorentz force"""

        # print(f' B field vector values: {B} | B field strength : {self.field.field_strength(B)}')

        x = state['x']
        y = state['y']
        tx = state['tx']
        ty = state['ty']
        q_over_p = state['q/p']
        dx = tx
        dy = ty
        dtx = q_over_p * np.sqrt(1 + tx**2 + ty**2) * (ty*(tx*B[0] + B[2]) - (1 + tx**2)*B[1])
        dty = -q_over_p * np.sqrt(1 + tx**2 + ty**2) * (tx*(ty*B[1] + B[2]) - (1 + ty**2)*B[0])
        return {'x': dx, 'y': dy, 'z' : 0, 'tx': dtx, 'ty': dty, 'q/p': 0}

    def rk4_step(self,z):
        """Perform a single RK4 step to update the particles' positions and momenta"""
        for particle in self.particles:
            state = particle.get_state()
            print(f'state : {state}')
            # Compute k1
            k1 = self.compute_state_derivative(state,self.field.interpolated_field(state['x'], state['y'], state['z']))

            # Compute k2
            state_k2 = {key: state[key] + 0.5 * self.dz * k1[key] for key in state}
            k2 = self.compute_state_derivative(state_k2,self.field.interpolated_field(state['x'], state['y'], state['z'] + 0.5 * self.dz))

            # Compute k3
            state_k3 = {key: state[key] + 0.5 * self.dz * k2[key] for key in state}
            k3 = self.compute_state_derivative(state_k3, self.field.interpolated_field(state['x'], state['y'],state['z'] + 0.5 * self.dz))

            # Compute k4
            state_k4 = {key: state[key] + self.dz * k3[key] for key in state}
            k4 = self.compute_state_derivative(state_k4, self.field.interpolated_field(state['x'], state['y'],state['z'] + self.dz))

            # Update state
            for key in state:
                state[key] += (self.dz / 6.0) * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
            state['z'] = self.z + self.dz
            particle.update_state(state)
            particle.record_state()


    def run(self):
        """Run the simulation for the specified number of steps"""
        i = 0
        for _ in range(self.num_steps):
            self.rk4_step(self.z)
            self.z += self.dz
        for particle in self.particles:
            particle.end_run(self.output_dir)


    def plot_trajectory_with_lorentz_force(self):
        """Plot the particles' 3D trajectories with arrows for Lorentz force"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        lines = []
        labels = []

        for particle in self.particles:
            states = particle.get_state_histores()
            x = [state['x'] for state in states]
            y = [state['y'] for state in states]
            z = [state['z'] for state in states]

            # Calculate Lorentz forces at each point
            lorentz_forces = [self.field.interpolated_field(state['x'], state['y'], state['z']) for state in states]
            force_magnitudes = [self.field.field_strength(force) for force in lorentz_forces]

            # Plot the particle trajectory with Z-axis as horizontal
            line, = ax.plot(z, y, x, label=f"{particle.Ptype} trajectory", lw=2)
            lines.append(line)
            labels.append(f"{particle.Ptype} trajectory")

            # # Plot arrows for the Lorentz force at every nth point
            # n = self.num_steps // 50
            # ax.quiver(x[::n], y[::n], [force[0] for force in lorentz_forces[::n]],
            #           [force[1] for force in lorentz_forces[::n]], angles='xy', scale_units='xy', scale=1, color='red')
            # ax.quiver(x[::n], y[::n], [state['tx'] for state in states[::n]],
            #           [state['ty'] for state in states[::n]], angles='xy', scale_units='xy', scale=1, color='green')

        # Set labels and show plot
        ax.set_xlabel('Z position (cm)')
        ax.set_ylabel('Y position (cm)')
        ax.set_zlabel('X position (cm)')
        ax.set_title('dz Sim')
        ax.legend()

        # Create check buttons for toggling visibility
        rax = plt.axes([0.05, 0.4, 0.15, 0.15])
        check = CheckButtons(rax, labels, [True] * len(labels))

        def toggle_visibility(label):
            index = labels.index(label)
            lines[index].set_visible(not lines[index].get_visible())
            plt.draw()

        check.on_clicked(toggle_visibility)

        # plt.show()


class dz_propagator():

    def __init__(self, particle_state, field: MagneticField, dz, z):
        self.particle = particle_state  # List of particles, using the particle_state object
        self.field = field
        self.dz = dz
        self.z = z

    def compute_state_derivative(self, state, B):
        """Compute the derivative of the state vector using the Lorentz force"""

        # print(f' B field vector values: {B} | B field strength : {self.field.field_strength(B)}')

        x = state['x']
        y = state['y']
        tx = state['tx']
        ty = state['ty']
        q_over_p = state['q/p']
        dx = tx
        dy = ty
        dtx = q_over_p * np.sqrt(1 + tx**2 + ty**2) * (ty*(tx*B[0] + B[2]) - (1 + tx**2)*B[1])
        dty = -q_over_p * np.sqrt(1 + tx**2 + ty**2) * (tx*(ty*B[1] + B[2]) - (1 + ty**2)*B[0])
        return {'x': dx, 'y': dy, 'z' : 0, 'tx': dtx, 'ty': dty, 'q/p': 0}

    def rk4_step(self,z):
        """Perform a single RK4 step to update the particles' positions and momenta"""
        state = self.particle.get_state()
        print(f'state : {state}')
        # Compute k1
        k1 = self.compute_state_derivative(state,self.field.interpolated_field(state['x'], state['y'], state['z']))

        # Compute k2
        state_k2 = {key: state[key] + 0.5 * self.dz * k1[key] for key in state}
        k2 = self.compute_state_derivative(state_k2,self.field.interpolated_field(state['x'], state['y'], state['z'] + 0.5 * self.dz))

        # Compute k3
        state_k3 = {key: state[key] + 0.5 * self.dz * k2[key] for key in state}
        k3 = self.compute_state_derivative(state_k3, self.field.interpolated_field(state['x'], state['y'],state['z'] + 0.5 * self.dz))

        # Compute k4
        state_k4 = {key: state[key] + self.dz * k3[key] for key in state}
        k4 = self.compute_state_derivative(state_k4, self.field.interpolated_field(state['x'], state['y'],state['z'] + self.dz))

        # Update state
        for key in state:
            state[key] += (self.dz / 6.0) * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
        state['z'] = self.z + self.dz


        return k1,k2,k3,k4
        


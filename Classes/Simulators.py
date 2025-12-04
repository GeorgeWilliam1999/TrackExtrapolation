# Simulators.py
"""
Runge-Kutta Simulators for Track Propagation.

This module provides classical RK4 integration for propagating charged
particle tracks through magnetic fields. The independent variable is z
(beam direction), not time.

Classes
-------
RK4_sim_dz : Multi-particle simulator with trajectory recording
dz_propagator : Single-particle single-step propagator

The state derivative equations implement the Lorentz force in the
LHCb parameterization:
- dx/dz = tx
- dy/dz = ty
- dtx/dz = q/p * gamma * (ty*(tx*Bx + Bz) - (1 + tx²)*By)
- dty/dz = -q/p * gamma * (tx*(ty*By + Bz) - (1 + ty²)*Bx)

where gamma = sqrt(1 + tx² + ty²).

Example
-------
>>> from Classes.Simulators import RK4_sim_dz
>>> from Classes.magnetic_field import LHCb_Field
>>> from Classes.particle import particle_state
>>>
>>> field = LHCb_Field("Data/Bfield.rtf")
>>> particle = particle_state("muon", [0, 0, 0], 0.1, 0.05, [0, 0, 10e9], -1)
>>> sim = RK4_sim_dz([particle], field, dz=10, z=0, num_steps=100)
>>> sim.run()
>>> sim.plot_trajectory_with_lorentz_force()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import os
import datetime

from Classes.magnetic_field import MagneticField, Quadratic_Field, LHCb_Field


class RK4_sim_dz:
    """
    Multi-particle RK4 simulator with trajectory recording.

    Propagates multiple particles through a magnetic field using classical
    4th-order Runge-Kutta integration in z (beam direction).

    Parameters
    ----------
    particle_states : list
        List of particle_state objects to propagate.
    field : MagneticField
        Magnetic field object with interpolated_field method.
    dz : float
        Step size in z (cm).
    z : float
        Initial z position (cm).
    num_steps : int
        Number of integration steps.

    Attributes
    ----------
    particles : list
        List of particle_state objects.
    field : MagneticField
        Magnetic field interpolator.
    dz : float
        Integration step size.
    z : float
        Current z position.
    num_steps : int
        Total number of steps.
    output_dir : str
        Directory for trajectory JSON output.
    """

    def __init__(self, particle_states, field: MagneticField, dz, z, num_steps):
        self.particles = particle_states
        self.field = field
        self.dz = dz
        self.z = z
        self.num_steps = num_steps
        self.output_dir = "Recorded_trajectories/run_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_state_derivative(self, state, B):
        """
        Compute state derivatives using Lorentz force equation.

        Parameters
        ----------
        state : dict
            Current state with keys 'x', 'y', 'z', 'tx', 'ty', 'q/p'.
        B : array-like
            Magnetic field vector [Bx, By, Bz] at current position.

        Returns
        -------
        dict
            State derivatives with same keys as input state.
        """
        tx = state['tx']
        ty = state['ty']
        q_over_p = state['q/p']

        gamma = np.sqrt(1 + tx**2 + ty**2)

        dx = tx
        dy = ty
        dtx = q_over_p * gamma * (ty * (tx * B[0] + B[2]) - (1 + tx**2) * B[1])
        dty = -q_over_p * gamma * (tx * (ty * B[1] + B[2]) - (1 + ty**2) * B[0])

        return {'x': dx, 'y': dy, 'z': 0, 'tx': dtx, 'ty': dty, 'q/p': 0}

    def rk4_step(self, z):
        """
        Perform single RK4 step for all particles.

        Parameters
        ----------
        z : float
            Current z position (for field evaluation).
        """
        for particle in self.particles:
            state = particle.get_state()
            print(f'state : {state}')

            # k1: derivative at current state
            k1 = self.compute_state_derivative(
                state,
                self.field.interpolated_field(state['x'], state['y'], state['z'])
            )

            # k2: derivative at half-step
            state_k2 = {key: state[key] + 0.5 * self.dz * k1[key] for key in state}
            k2 = self.compute_state_derivative(
                state_k2,
                self.field.interpolated_field(state['x'], state['y'], state['z'] + 0.5 * self.dz)
            )

            # k3: derivative at half-step with k2
            state_k3 = {key: state[key] + 0.5 * self.dz * k2[key] for key in state}
            k3 = self.compute_state_derivative(
                state_k3,
                self.field.interpolated_field(state['x'], state['y'], state['z'] + 0.5 * self.dz)
            )

            # k4: derivative at full step
            state_k4 = {key: state[key] + self.dz * k3[key] for key in state}
            k4 = self.compute_state_derivative(
                state_k4,
                self.field.interpolated_field(state['x'], state['y'], state['z'] + self.dz)
            )

            # Update state with weighted average
            for key in state:
                state[key] += (self.dz / 6.0) * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
            state['z'] = self.z + self.dz

            particle.update_state(state)
            particle.record_state()

    def run(self):
        """
        Run simulation for all particles over num_steps.

        Saves trajectory JSON files to output_dir.
        """
        for _ in range(self.num_steps):
            self.rk4_step(self.z)
            self.z += self.dz

        for particle in self.particles:
            particle.end_run(self.output_dir)


    def plot_trajectory_with_lorentz_force(self):
        """
        Plot 3D particle trajectories with interactive visibility toggle.

        Creates a 3D plot showing all particle trajectories with
        check buttons for toggling individual trajectory visibility.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        lines = []
        labels = []

        for particle in self.particles:
            states = particle.get_state_histores()
            x = [state['x'] for state in states]
            y = [state['y'] for state in states]
            z = [state['z'] for state in states]

            # Plot trajectory (z as horizontal axis)
            line, = ax.plot(z, y, x, label=f"{particle.Ptype} trajectory", lw=2)
            lines.append(line)
            labels.append(f"{particle.Ptype} trajectory")

        ax.set_xlabel('Z position (cm)')
        ax.set_ylabel('Y position (cm)')
        ax.set_zlabel('X position (cm)')
        ax.set_title('dz Sim')
        ax.legend()

        # Toggle visibility buttons
        rax = plt.axes([0.05, 0.4, 0.15, 0.15])
        check = CheckButtons(rax, labels, [True] * len(labels))

        def toggle_visibility(label):
            index = labels.index(label)
            lines[index].set_visible(not lines[index].get_visible())
            plt.draw()

        check.on_clicked(toggle_visibility)


class dz_propagator:
    """
    Single-particle single-step RK4 propagator.

    Performs one RK4 step for a single particle, returning
    the intermediate stage derivatives (k1, k2, k3, k4).

    Parameters
    ----------
    particle_state : particle_state
        Particle to propagate.
    field : MagneticField
        Magnetic field interpolator.
    dz : float
        Step size in z.
    z : float
        Current z position.

    Attributes
    ----------
    particle : particle_state
        The particle being propagated.
    field : MagneticField
        Magnetic field object.
    dz : float
        Step size.
    z : float
        Current position.
    """

    def __init__(self, particle_state, field: MagneticField, dz, z):
        self.particle = particle_state
        self.field = field
        self.dz = dz
        self.z = z

    def compute_state_derivative(self, state, B):
        """
        Compute state derivatives using Lorentz force.

        Parameters
        ----------
        state : dict
            Particle state dictionary.
        B : array-like
            Magnetic field vector [Bx, By, Bz].

        Returns
        -------
        dict
            State derivatives.
        """
        tx = state['tx']
        ty = state['ty']
        q_over_p = state['q/p']

        gamma = np.sqrt(1 + tx**2 + ty**2)

        dx = tx
        dy = ty
        dtx = q_over_p * gamma * (ty * (tx * B[0] + B[2]) - (1 + tx**2) * B[1])
        dty = -q_over_p * gamma * (tx * (ty * B[1] + B[2]) - (1 + ty**2) * B[0])

        return {'x': dx, 'y': dy, 'z': 0, 'tx': dtx, 'ty': dty, 'q/p': 0}

    def rk4_step(self, z):
        """
        Perform single RK4 step and return stage derivatives.

        Parameters
        ----------
        z : float
            Current z position.

        Returns
        -------
        tuple
            (k1, k2, k3, k4) stage derivative dictionaries.
        """
        state = self.particle.get_state()
        print(f'state : {state}')

        k1 = self.compute_state_derivative(
            state,
            self.field.interpolated_field(state['x'], state['y'], state['z'])
        )

        state_k2 = {key: state[key] + 0.5 * self.dz * k1[key] for key in state}
        k2 = self.compute_state_derivative(
            state_k2,
            self.field.interpolated_field(state['x'], state['y'], state['z'] + 0.5 * self.dz)
        )

        state_k3 = {key: state[key] + 0.5 * self.dz * k2[key] for key in state}
        k3 = self.compute_state_derivative(
            state_k3,
            self.field.interpolated_field(state['x'], state['y'], state['z'] + 0.5 * self.dz)
        )

        state_k4 = {key: state[key] + self.dz * k3[key] for key in state}
        k4 = self.compute_state_derivative(
            state_k4,
            self.field.interpolated_field(state['x'], state['y'], state['z'] + self.dz)
        )

        # Update state
        for key in state:
            state[key] += (self.dz / 6.0) * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
        state['z'] = self.z + self.dz

        return k1, k2, k3, k4
        


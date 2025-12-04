# particle.py
"""
Particle State Representation for Track Extrapolation.

This module provides the ``particle_state`` class for representing
and tracking charged particle states through a magnetic field.

The particle state vector consists of:
- Position: (x, y, z) in cm
- Direction tangents: tx = dx/dz, ty = dy/dz
- Charge over momentum: q/p in c/GeV

Example
-------
>>> from Classes.particle import particle_state
>>> particle = particle_state(
...     Ptype="muon",
...     position=[0, 0, 0],
...     tx=0.1, ty=0.05,
...     momentum=[0, 0, 10e9],  # 10 GeV
...     charge=-1
... )
>>> print(particle.get_state())
"""

import numpy as np
import json
import datetime
import os


class particle_state:
    """
    Representation of a charged particle state for tracking.

    Stores the particle type and 6D state vector, and provides methods
    for state updates and trajectory recording.

    Parameters
    ----------
    Ptype : str
        Particle type identifier (e.g., "muon", "pion", "electron").
    position : array-like
        Initial position [x, y, z] in cm.
    tx : float
        Direction tangent dx/dz (dimensionless).
    ty : float
        Direction tangent dy/dz (dimensionless).
    momentum : array-like
        Momentum vector [px, py, pz] in MeV/c (only magnitude used).
    charge : int
        Particle charge in units of e (+1, -1, etc.).

    Attributes
    ----------
    Ptype : str
        Particle type identifier.
    state : dict
        Current state with keys: 'x', 'y', 'z', 'tx', 'ty', 'q/p'.
    state_histores : list
        History of all recorded states.

    Examples
    --------
    >>> p = particle_state("muon", [0, 0, 0], 0.1, 0.0, [0, 0, 10e9], -1)
    >>> state = p.get_state()
    >>> print(f"q/p = {state['q/p']:.6e}")
    """

    def __init__(self, Ptype, position, tx, ty, momentum, charge):
        self.Ptype = Ptype
        self.state = {
            'x': position[0],
            'y': position[1],
            'z': position[2],
            'tx': tx,
            'ty': ty,
            'q/p': charge / np.linalg.norm(momentum)
        }
        self.state_histores = [self.state.copy()]
        self.record_state()
        print(f'init state : {self.state}')

    def update_state(self, state):
        """
        Update the particle state.

        Parameters
        ----------
        state : dict
            New state dictionary with keys 'x', 'y', 'z', 'tx', 'ty', 'q/p'.
        """
        self.state = state

    def record_state(self):
        """Append current state to history."""
        self.state_histores.append(self.state.copy())

    def get_state(self):
        """
        Get current particle state.

        Returns
        -------
        dict
            State dictionary with keys 'x', 'y', 'z', 'tx', 'ty', 'q/p'.
        """
        return self.state

    def get_state_histores(self):
        """
        Get full state history.

        Returns
        -------
        list
            List of state dictionaries from trajectory.
        """
        return self.state_histores

    def end_run(self, output_dir):
        """
        Save trajectory history to JSON file.

        Parameters
        ----------
        output_dir : str
            Directory path for output file.
        """
        filename = os.path.join(output_dir, f"{self.Ptype}.json")
        with open(filename, "w") as f:
            json.dump(self.state_histores, f, indent=4)

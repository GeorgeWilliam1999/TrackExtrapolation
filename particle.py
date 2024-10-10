# particle.py

import numpy as np

class Particle:
    def __init__(self, charge, mass, position, velocity):
        self.charge = charge
        self.mass = mass
        self.position = np.array(position)  # [x, y, z]
        self.velocity = np.array(velocity)  # [vx, vy, vz]
        self.positions = [np.array(position)]  # To store the trajectory
        self.velocities = [np.array(velocity)]

    def update_position(self, new_position):
        self.position = new_position
        self.positions.append(new_position)

    def update_velocity(self, new_velocity):
        self.velocity = new_velocity
        self.velocities.append(new_velocity)
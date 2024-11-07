# particle.py

import numpy as np

class Particle:
    def __init__(self,Ptype, charge, mass, position, momentum):
        self.Ptype = Ptype
        self.charge = charge
        self.mass = mass
        self.position = np.array(position)
        self.momentum = np.array(momentum)
        self.positions = [self.position.copy()]
        self.momenta = [self.momentum.copy()]


    def update_momentum(self, momentum):
        self.momentum = momentum
        self.momenta.append(momentum.copy())

    def update_position(self, position):
        self.position = position
        self.positions.append(position.copy())

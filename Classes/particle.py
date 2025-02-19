# particle.py

import numpy as np
import json, datetime
import os

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

class particle_state:

    def __init__(self, Ptype, position, tx, ty, momentum, charge):
        self.Ptype = Ptype
        self.state = {'x' : position[0], 'y' : position[1], 'z' : position[2], 'tx' : tx, 'ty' : ty, 'q/p' : charge/np.linalg.norm(momentum)}
        self.state_histores = [self.state.copy()]
        self.record_state()

        print(f'init state : {self.state}')
    def update_state(self, state):
        self.state = state

    def record_state(self):
        self.state_histores.append(self.state.copy())

    def get_state(self):
        return self.state

    def get_state_histores(self):
        return self.state_histores
    
    def end_run(self,output_dir):
        filename = os.path.join(output_dir, f"{self.Ptype}.json")
        with open(filename, "w") as f:
            json.dump(self.state_histores, f, indent=4)

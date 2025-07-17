# main_script.py

from Classes.magnetic_field import Quadratic_Field, LHCb_Field
from Classes.particle import  particle_state
from Classes.Simulators import RK4_sim_dz

import numpy as np
import random
import os
import datetime

import matplotlib.pyplot as plt

############## SIMULATION PARAMETERS ##############

data = 'Data/twodip.rtf'

num_steps = 100  # Number of steps in the simulation


# Natural units: c = 1, ħ = 1
# Charge of electron in natural units (e = 1)
e_charge = -1.0
# Mass of electron in natural units (me = 511 keV/c^2)
e_mass = 511e3  # in eV

# Charge and mass of proton in natural units
p_charge = 1.0
p_mass = 938.272e6  # in eV

# Charge and mass of neutron in natural units
n_charge = 0.0
n_mass = 939.565e6  # in eV

# Mass of pion in natural units
pi_mass = 139.57018e6  # in eV

# Mass of muon in natural units
mu_mass = 105.6583745e6  # in eV

# field = LHCb_Field('Data/Bfield.rtf')
Qfield = Quadratic_Field(1e-7)
LHCbField = LHCb_Field(data)

############## PARTICLES STATES for dz SIM ##############

particle_states = [
    particle_state(Ptype='Electron', position=[200, 200, 0], tx=0, ty=0, momentum=[0, 0, 0.99 * e_mass], charge=e_charge),
    particle_state(Ptype='Positron', position=[210, 210, 0], tx=0, ty=0, momentum=[0, 0, 0.99 * e_mass], charge=p_charge),
    particle_state(Ptype='Proton', position=[200, 200, 0], tx=0, ty=0, momentum=[0, 0, 0.9 * p_mass], charge=p_charge),
    particle_state(Ptype='Neutron', position=[200, 200, 0], tx=0, ty=0, momentum=[0, 0, 0.9 * n_mass], charge=n_charge),
    particle_state(Ptype='Pion', position=[200, 200, 0], tx=0, ty=0, momentum=[0, 0, 0.9 * pi_mass], charge=p_charge),
    particle_state(Ptype='Muon', position=[200, 200, 0], tx=0, ty=0, momentum=[0, 0, 0.99 * mu_mass], charge=e_charge),
   ]

random_states = []
for i in range(10):
    x = random.uniform(-4000, 4000)
    y = random.uniform(-4000, 4000)
    random_states.append(
        particle_state(
            Ptype=str(i),
            position=[x, y, -500],
            tx=0,
            ty=0,
            momentum=[0, 0, 0.89 * p_mass],
            charge=random.randint(-1, 1)
        )
    )

############## SIMULATION FUNCTIONS ##############

def RK4_simulate_particles_dz(particles, field, dz, z, num_steps):

    print('INITIALIZING SIMULATION : dz...')

    # Create a simulation for the particles
    simulation = RK4_sim_dz(particles, field, dz, z, num_steps)

    print('RUNNING SIMULATION...')

    # Run simulation
    simulation.run()

    print('SIMULATION COMPLETE...')

    # Plot the trajectories with the magnetic field
    simulation.plot_trajectory_with_lorentz_force()


############## RUN SIMULATIONS ##############

def run_all_simulations(num_steps):
    
    # RK4_simulate_particles_dt(LHCbField, particles_all, dt, num_steps)
    RK4_simulate_particles_dz(random_states, Qfield, 14500/num_steps, -500, num_steps)


    print('PLOTTING SIMULATIONS...')

    plt.show()




run_all_simulations(num_steps)

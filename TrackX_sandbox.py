# main_script.py

from Classes.magnetic_field import Quadratic_Field, LHCb_Field
from Classes.particle import Particle, particle_state
from Classes.Simulators import RK4_sim_dt, RK4_sim_dz

import matplotlib.pyplot as plt

############## SIMULATION PARAMETERS ##############

data = 'Data/Bfield_recentred.rtf'
dt = 1e-1     # Time step (seconds)
num_steps = 10  # Number of steps in the simulation

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
Qfield = Quadratic_Field(1e-3)
LHCbField = LHCb_Field(data)

############## PARTICLES for dt SIM ##############

particles_all = [
    Particle(Ptype='Electron', charge=e_charge, mass=e_mass, position=[200, 200, 0], momentum=[0, 0, 0.99 * e_mass]),
    # Particle(Ptype='Positrom', charge=e_charge, mass=e_mass, position=[210, 210, 0], momentum=[0, 0, 0.9 * e_mass]),
    # # Particle(Ptype='Proton', charge=p_charge, mass=p_mass, position=[200, 200, 0], momentum=[0, 0, 0.9 * p_mass]),
    # # Particle(Ptype='Neutron', charge=n_charge, mass=n_mass, position=[200, 200, 0], momentum=[0, 0, 0.9 * n_mass]),
    # # Particle(Ptype='Pion', charge=p_charge, mass=pi_mass, position=[200, 200, 0], momentum=[0, 0, 0.9 * pi_mass]),
    # Particle(Ptype='Muon Fast', charge=e_charge, mass=mu_mass, position=[190, 190, 0], momentum=[0, 0, 0.99 * mu_mass]),
    # Particle(Ptype='Muon Slow', charge=e_charge, mass=mu_mass, position=[200, 200, 0], momentum=[0, 0, 0.9 * mu_mass])
]

particles_electrons = [
    Particle(Ptype='Positron 1', charge=p_charge, mass=e_mass, position=[20, 20, 0], momentum=[0, 0, 0.99 * e_mass]),
    Particle(Ptype='Electron 2', charge=e_charge, mass=e_mass, position=[20, 20, 0], momentum=[0, 0, 0.99 * e_mass]),
    Particle(Ptype='Electron 3', charge=e_charge, mass=e_mass, position=[20, 20, 0], momentum=[0, 0, 0.99 * e_mass])
]

############## PARTICLES STATES for dz SIM ##############

particle_state_electron = [
    particle_state(Ptype='Electron', position=[200, 200, 0], tx=0, ty=0, momentum=[0, 0, 0.99 * e_mass], charge=e_charge),
    # particle_state(Ptype='Positron', position=[210, 210, 0], tx=0, ty=0, momentum=[0, 0, 0.99 * e_mass], charge=p_charge),
    # particle_state(Ptype='Proton', position=[40, 40, 0], tx=0, ty=0, momentum=[0, 0, 0.9 * p_mass], charge=p_charge),
    # particle_state(Ptype='Neutron', position=[50, 50, 0], tx=0, ty=0, momentum=[0, 0, 0.9 * n_mass], charge=n_charge),
    # particle_state(Ptype='Pion', position=[60, 60, 0], tx=0, ty=0, momentum=[0, 0, 0.9 * pi_mass], charge=p_charge),
    # particle_state(Ptype='Muon', position=[190, 190, 0], tx=0, ty=0, momentum=[0, 0, 0.99 * mu_mass], charge=e_charge),
   ]


############## SIMULATION FUNCTIONS ##############

def RK4_simulate_particles_dt(field, particles, dt, num_steps):

    print('INITIALIZING SIMULATION : dt...')

    # Create a simulation for the particles
    simulation = RK4_sim_dt(particles, field, dt, num_steps)

    print('RUNNING SIMULATION...')

    # Run simulation
    simulation.run()

    print('SIMULATION COMPLETE...')

    # Plot the trajectories with the magnetic field
    simulation.plot_trajectory_with_lorentz_force()

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

def select_sim_type(sim_type,num_steps):
    if sim_type == 'dt':
        RK4_simulate_particles_dt(LHCbField, particles_all, dt, num_steps)
    elif sim_type == 'dz':
        RK4_simulate_particles_dz(particle_state_electron, LHCbField, 1400/num_steps, 0, num_steps)
    else:
        print('Invalid input. Please enter either "dt" or "dz".')

    plt.show()

def run_all_simulations(num_steps):
    RK4_simulate_particles_dt(LHCbField, particles_all, dt, num_steps)
    RK4_simulate_particles_dz(particle_state_electron, LHCbField, 1400/num_steps, 0, num_steps)

    print('PLOTTING SIMULATIONS...')

    plt.show()


run_all_simulations(1400)

# select_sim_type('dz',10000)

# select_sim_type('dt',1400)


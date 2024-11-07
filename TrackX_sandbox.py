# main_script.py

from Classes.magnetic_field import Quadratic_Field, LHCb_Field
from Classes.particle import Particle
from Classes.Simulators import RK4_sim

data = 'Data/Bfield_recentred.rtf'
dt = 1e0     # Time step (seconds)
num_steps = 1400  # Number of steps in the simulation

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

# field = LHCb_Field('Data/Bfield.rtf')
Qfield = Quadratic_Field(1e-3)
LHCbField = LHCb_Field(data)


particles_all = [
        Particle(Ptype = 'Electron', charge=e_charge, mass=e_mass, position=[20, 20, 0], momentum=[0, 0, 3/4 * 1e0 * e_mass]),
        Particle(Ptype = 'Proton', charge=p_charge, mass=p_mass, position=[30, 30, 0], momentum=[0, 0, 3/4 * 1e0 * p_mass]),
        Particle(Ptype = 'Neutron', charge=n_charge, mass=n_mass, position=[40, 40, 0], momentum=[0, 0, 3/4 * 1e0 * n_mass])
    ]

particles_electrons = [
        Particle(Ptype = 'Positron 1', charge=p_charge, mass=e_mass, position=[20, 20, 0], momentum=[0 * e_mass, 0 * e_mass, 3/4 * 1e0 * e_mass]),
        Particle(Ptype = 'Electron 2', charge=e_charge, mass=e_mass, position=[20, 20, 0], momentum=[0, -0.01 * e_mass, 3/4 * 1e0 * e_mass]),
        Particle(Ptype = 'Electron 3', charge=e_charge, mass=e_mass, position=[20, 20, 0], momentum=[-0.01 * e_mass, 0, 3/4 * 1e0 * e_mass])
        ]

def RK4_simulate_particles(field,particles,dt,num_steps):
    # Create a simulation for the particles
    simulation = RK4_sim(particles_electrons, field, dt, num_steps)

    # Run simulation
    simulation.run()

    # Plot the trajectories with the magnetic field
    simulation.plot_trajectory_with_lorentz_force()

RK4_simulate_particles(LHCbField,particles_all,dt,num_steps)

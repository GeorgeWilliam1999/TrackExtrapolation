# main_script.py

from magnetic_field import Quadratic_Field, LHCb_Field
from particle import Particle
from Simulators import RK4_sim

def RK4_simulate_particles_LHCb():

    data = 'Data/Bfield.rtf'
    dt = 1e-7     # Time step (seconds)
    num_steps = 100  # Number of steps in the simulation


    # Create a magnetic field
    magnetic_field = LHCb_Field(data)

    # Create a particle (charge, mass, initial position, initial velocity in the z-direction)
    electron = Particle(charge=-1.6e-19, mass=9.11e-31, position=[20, 20, 0], momentum=[0, 0, 1e1*9.11e-31])

    # Create a simulation for the particle
    electron_simulation = RK4_sim(electron, magnetic_field, dt, num_steps)

    # Run simulation
    electron_simulation.run()

    # Plot the trajectory with the magnetic field
    electron_simulation.plot_trajectory_with_lorentz_force()


def RK4_simulate_particles_quad():
    B0 = 1         # Base magnetic field strength (Tesla)
    dt = 1e-13     # Time step (seconds)
    num_steps = 1000  # Number of steps in the simulation

    # Create a magnetic field
    magnetic_field = Quadratic_Field(B0)

    # Create a particle (charge, mass, initial position, initial velocity in the z-direction)
    electron = Particle(charge=-1.6e-19, mass=9.11e-31, position=[20, 20, 0], momentum=[0, 0, 1e6/9.11e-31])

    # Create a simulation for the particle
    electron_simulation = RK4_sim(electron, magnetic_field, dt, num_steps)

    # Run simulation
    electron_simulation.run()

    # Plot the trajectory with the magnetic field
    electron_simulation.plot_trajectory_with_lorentz_force()

# Run the simulation
RK4_simulate_particles_LHCb()

# Run the simulation
# RK4_simulate_particles_quad()



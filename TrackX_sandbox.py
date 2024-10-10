# main_script.py

from magnetic_field import Quadratic_Field
from particle import Particle
from Simulators import RK4_sim

def RK4_simulate_particles():
    B0 = 10         # Base magnetic field strength (Tesla)
    dt = 1e-1     # Time step (seconds)
    num_steps = 1000  # Number of steps in the simulation

    # Create a magnetic field
    magnetic_field = Quadratic_Field(B0)

    # Create a particle (charge, mass, initial position, initial velocity in the z-direction)
    electron = Particle(charge=-1.6e-19, mass=9.11e-31, position=[0, 0, 0], velocity=[0, 0, 1e6])

    # Create a simulation for the particle
    electron_simulation = RK4_sim(electron, magnetic_field, dt, num_steps)

    # Run simulation
    electron_simulation.run()

    # Plot the trajectory with the magnetic field
    electron_simulation.plot_trajectory_with_lorentz_force()

# Run the simulation
RK4_simulate_particles()
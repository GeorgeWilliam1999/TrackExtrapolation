# magnetic_field.py

import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

class MagneticField(ABC):
    def __init__(self, B0 : None):
        self.B0 = B0  # Base magnetic field strength

    @abstractmethod
    def magnetic_field(self, x, y, z):
        """Calculate the magnetic field vector at (x, y, z)."""
        pass

    def field_strength(self, B):
        """Calculate the magnitude (absolute value) of the magnetic field vector B."""
        return np.linalg.norm(B)

    # def interpolated_field(self, x, y, z):
    #     """Interpolate the magnetic field vector at (x, y, z) using RegularGridInterpolator."""
    #     points = np.array([x, y, z]).T
    #     grid = np.array([np.arange(dim) for dim in self.Bfield.shape[:-1]])
    #     interpolator = RegularGridInterpolator(grid, self.Bfield)
    #     return interpolator(points)

# Concrete class inheriting from MagneticField
class QuadraticField(MagneticField):
    def magnetic_field(self, x, y, z):
        """Define a parabolic magnetic field Bx(x, y, z) = B0 * (x^2 - y^2),
        By(x, y, z) = B0 * (2xy), Bz = B0 * z"""
        Bx = self.B0 * (x**2 - y**2)  # Parabolic in X and Y
        By = self.B0 * (2 * x * y)     # Linear in XY
        Bz = self.B0 * z                # Linear in Z
        return np.array([Bx, By, Bz])

class Quadratic_Field:
    def __init__(self, B0):
        self.B0 = B0  # Base magnetic field strength

    def magnetic_field(self, x, y, z):
        """Define a parabolic magnetic field Bx(x, y) = B0 * (x^2 - y^2), By(x, y) = B0 * (2xy), Bz = B0 * z"""
        Bx = self.B0 * -4*z**2 - 4* z  # Parabolic in Z
        By = self.B0 * -4*z**2 - 4* z  # Parabolic in Z
        Bz = self.B0 * z               # Linear in z
        return np.array([Bx, By, Bz])

    def field_strength(self, B):
        """Calculate the magnitude (absolute value) of the magnetic field vector B"""
        return np.linalg.norm(B)

    def interpolated_field(self, x, y, z):
        """Interpolate the magnetic field vector at (x, y, z) using RegularGridInterpolator."""
        Bx = self.B0 * -4*z**2 - 4* z  # Parabolic in Z
        By = self.B0 * -4*z**2 - 4* z  # Parabolic in Z
        Bz = self.B0 * z               # Linear in z
        return np.array([Bx, By, Bz])



class LHCb_Field:
    def __init__(self, file_path):
        self.Bfield = self.load_Bfield(file_path)
        self.create_interpolators()
        # Define the grid points for x, y, z coordinates
        # self.x = np.arange(self.Bfield.shape[0])
        # self.y = np.arange(self.Bfield.shape[1])
        # self.z = np.arange(self.Bfield.shape[2])

    def load_Bfield(self, file_path):
        # Read the data from the file
        data = np.loadtxt(file_path)
        # Extract x, y, z, bx, by, bz columns
        x, y, z, bx, by, bz = data.T
        # Reshape the data into a grid
        self.x_grid = np.unique(x)
        self.y_grid = np.unique(y)
        self.z_grid = np.unique(z)
        self.bx = bx.reshape(len(self.x_grid), len(self.y_grid), len(self.z_grid))
        self.by = by.reshape(len(self.x_grid), len(self.y_grid), len(self.z_grid))
        self.bz = bz.reshape(len(self.x_grid), len(self.y_grid), len(self.z_grid))

    def field_strength(self, B):
        """Calculate the magnitude (absolute value) of the magnetic field vector B"""
        return np.linalg.norm(B)

    def create_interpolators(self):
        # Create interpolators for each component of the B field
        self.interpolator_bx = RegularGridInterpolator((self.x_grid, self.y_grid, self.z_grid), self.bx)
        self.interpolator_by = RegularGridInterpolator((self.x_grid, self.y_grid, self.z_grid), self.by)
        self.interpolator_bz = RegularGridInterpolator((self.x_grid, self.y_grid, self.z_grid), self.bz)

    def interpolated_field(self, x, y, z):
        """Interpolate the magnetic field vector at (x, y, z) using RegularGridInterpolator."""
        point = np.array([x, y, z])

                # Check if the point is within bounds
        if (point[0] < self.x_grid[0] or point[0] > self.x_grid[-1] or
            point[1] < self.y_grid[0] or point[1] > self.y_grid[-1] or
            point[2] < self.z_grid[0] or point[2] > self.z_grid[-1]):
            return np.array([0.0, 0.0, 0.0]).reshape(3,)


        bx = self.interpolator_bx(point)
        by = self.interpolator_by(point)
        bz = self.interpolator_bz(point)
        return np.array([bx, by, bz]).reshape(3,)

    def plot_interpolated_field(self, x, y, z_values):
        """Plot the interpolated magnetic field for different values of z."""
        Bx, By, Bz = [], [], []
        for z in z_values:
            B = self.interpolated_field(x, y, z)
            Bx.append(B[0])
            By.append(B[1])
            Bz.append(B[2])

        plt.figure(figsize=(10, 6))
        plt.plot(z_values, Bx, label='Bx')
        plt.plot(z_values, By, label='By')
        plt.plot(z_values, Bz, label='Bz')
        plt.xlabel('z')
        plt.ylabel('Magnetic Field (T)')
        plt.title('Interpolated Magnetic Field Components vs z')
        plt.legend()
        plt.grid(True)
        plt.show()
# magnetic_field.py

import numpy as np
from abc import ABC, abstractmethod

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


class LHCb_Field:
    def __init__(self,data):
        self.B0 = 1e-4
        field = np.loadtxt(data)
        for i in [0,1,2]:
            field[:,i] += np.abs(np.min(field[:,i]))
            field[:,i] /= 10

        self.Bfield = np.zeros((41,41,146,3))
        for r in field:

            self.Bfield[int(r[0]),int(r[1]),int(r[2])] = r[3:] * self.B0

    def magnetic_field(self, x, y, z):
        # Round the input coordinates to the nearest integer
        x_idx, y_idx, z_idx = round(x), round(y), round(z)

        # Get the dimensions of the Bfield array
        max_x, max_y, max_z, _ = self.Bfield.shape

        # Check if the rounded indices are within the valid range
        if 0 <= x_idx < max_x and 0 <= y_idx < max_y and 0 <= z_idx < max_z:
            # Valid indices, return the magnetic field at this point
            return self.Bfield[x_idx, y_idx, z_idx]
        else:
            # Invalid indices, return (0, 0, 0)
            return (0, 0, 0)

    def field_strength(self, B):
        """Calculate the magnitude (absolute value) of the magnetic field vector B."""
        return np.linalg.norm(B)


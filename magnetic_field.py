# magnetic_field.py

import numpy as np

class Quadratic_Field:
    def __init__(self, B0):
        self.B0 = B0  # Base magnetic field strength

    def parabolic_field(self, x, y, z):
        """Define a parabolic magnetic field Bx(x, y) = B0 * (x^2 - y^2), By(x, y) = B0 * (2xy), Bz = B0 * z"""
        Bx = self.B0 * -4*z**2 - 4* z  # Parabolic in Z
        By = self.B0 * -4*z**2 - 4* z  # Parabolic in Z
        Bz = self.B0 * z * 0           # Linear in z
        return np.array([Bx, By, Bz])

    def field_strength(self, B):
        """Calculate the magnitude (absolute value) of the magnetic field vector B"""
        return np.linalg.norm(B)

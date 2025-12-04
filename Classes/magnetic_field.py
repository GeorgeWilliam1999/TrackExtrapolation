# magnetic_field.py
"""
Magnetic Field Implementations for Track Extrapolation.

This module provides magnetic field classes used in particle tracking simulations.
The main classes are:

- ``MagneticField``: Abstract base class defining the magnetic field interface
- ``Quadratic_Field``: Simple analytical field for testing purposes
- ``LHCb_Field``: Real LHCb magnetic field data with trilinear interpolation

Example
-------
>>> from Classes.magnetic_field import LHCb_Field
>>> field = LHCb_Field("path/to/Bfield.rtf")
>>> B = field.interpolated_field(x=0, y=0, z=500)
>>> print(B)  # [Bx, By, Bz] in Tesla

Notes
-----
The LHCb field data is expected in a text file with columns:
x, y, z, Bx, By, Bz (space-separated).
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


class MagneticField(ABC):
    """
    Abstract base class for magnetic field implementations.

    This class defines the interface that all magnetic field implementations
    must follow. Subclasses must implement the ``magnetic_field`` method.

    Parameters
    ----------
    B0 : float, optional
        Base magnetic field strength (used by some subclasses).

    Attributes
    ----------
    B0 : float
        Stored base field strength.

    See Also
    --------
    Quadratic_Field : Simple analytical field implementation.
    LHCb_Field : Real LHCb detector field with interpolation.
    """

    def __init__(self, B0=None):
        self.B0 = B0

    @abstractmethod
    def magnetic_field(self, x, y, z):
        """
        Calculate the magnetic field vector at position (x, y, z).

        Parameters
        ----------
        x : float
            X-coordinate in cm.
        y : float
            Y-coordinate in cm.
        z : float
            Z-coordinate in cm.

        Returns
        -------
        numpy.ndarray
            Magnetic field vector [Bx, By, Bz] in Tesla.
        """
        pass

    def field_strength(self, B):
        """
        Calculate the magnitude of a magnetic field vector.

        Parameters
        ----------
        B : array-like
            Magnetic field vector [Bx, By, Bz].

        Returns
        -------
        float
            Magnitude |B| = sqrt(Bx² + By² + Bz²).
        """
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
    """
    Analytical quadratic magnetic field for testing.

    This class provides a simple analytical magnetic field with quadratic
    dependence on the z-coordinate. Useful for testing track propagation
    algorithms without loading external data.

    Parameters
    ----------
    B0 : float
        Base magnetic field strength scaling factor.

    Examples
    --------
    >>> field = Quadratic_Field(B0=1e-3)
    >>> B = field.interpolated_field(0, 0, 100)
    >>> print(B)  # [Bx, By, Bz]
    """

    def __init__(self, B0):
        self.B0 = B0

    def magnetic_field(self, x, y, z):
        """
        Calculate quadratic magnetic field at position (x, y, z).

        The field profile is:
        - Bx = B0 * (-4z² - 4z)
        - By = B0 * (-4z² - 4z)
        - Bz = B0 * z

        Parameters
        ----------
        x : float
            X-coordinate (unused in this model).
        y : float
            Y-coordinate (unused in this model).
        z : float
            Z-coordinate in cm.

        Returns
        -------
        numpy.ndarray
            Field vector [Bx, By, Bz].
        """
        Bx = self.B0 * (-4 * z**2 - 4 * z)
        By = self.B0 * (-4 * z**2 - 4 * z)
        Bz = self.B0 * z
        return np.array([Bx, By, Bz])

    def field_strength(self, B):
        """
        Calculate magnitude of field vector.

        Parameters
        ----------
        B : array-like
            Field vector [Bx, By, Bz].

        Returns
        -------
        float
            Field magnitude |B|.
        """
        return np.linalg.norm(B)

    def interpolated_field(self, x, y, z):
        """
        Get field at position (convenience wrapper for magnetic_field).

        Parameters
        ----------
        x, y, z : float
            Position coordinates.

        Returns
        -------
        numpy.ndarray
            Field vector [Bx, By, Bz].
        """
        return self.magnetic_field(x, y, z)



class LHCb_Field:
    """
    LHCb magnetic field with trilinear interpolation.

    Loads field map data from a text file and provides interpolated
    magnetic field values at arbitrary positions within the field volume.
    Uses scipy's RegularGridInterpolator for efficient trilinear interpolation.

    Parameters
    ----------
    file_path : str
        Path to the field map file. Expected format is space-separated columns:
        x, y, z, Bx, By, Bz

    Attributes
    ----------
    x_grid, y_grid, z_grid : numpy.ndarray
        1D arrays of unique grid coordinates.
    bx, by, bz : numpy.ndarray
        3D arrays of field components on the grid.
    interpolator_bx, interpolator_by, interpolator_bz : RegularGridInterpolator
        Interpolators for each field component.

    Examples
    --------
    >>> field = LHCb_Field("Data/Bfield.rtf")
    >>> B = field.interpolated_field(0, 0, 500)
    >>> print(f"Field strength: {field.field_strength(B):.4f} T")

    Notes
    -----
    Points outside the grid bounds return zero field [0, 0, 0].
    """

    def __init__(self, file_path):
        self.Bfield = self.load_Bfield(file_path)
        self.create_interpolators()

    def load_Bfield(self, file_path):
        """
        Load magnetic field data from text file.

        Parameters
        ----------
        file_path : str
            Path to field data file.

        Returns
        -------
        None
            Sets instance attributes x_grid, y_grid, z_grid, bx, by, bz.
        """
        data = np.loadtxt(file_path)
        x, y, z, bx, by, bz = data.T

        self.x_grid = np.unique(x)
        self.y_grid = np.unique(y)
        self.z_grid = np.unique(z)
        self.bx = bx.reshape(len(self.x_grid), len(self.y_grid), len(self.z_grid))
        self.by = by.reshape(len(self.x_grid), len(self.y_grid), len(self.z_grid))
        self.bz = bz.reshape(len(self.x_grid), len(self.y_grid), len(self.z_grid))

    def field_strength(self, B):
        """
        Calculate field magnitude with scaling factor.

        Parameters
        ----------
        B : array-like
            Field vector [Bx, By, Bz].

        Returns
        -------
        float
            Scaled field magnitude.

        Notes
        -----
        The 1e13 scaling factor is for visualization/debugging.
        For physics calculations, use np.linalg.norm(B) directly.
        """
        return np.linalg.norm(B) * 1e13

    def create_interpolators(self):
        """Create scipy RegularGridInterpolators for each field component."""
        self.interpolator_bx = RegularGridInterpolator(
            (self.x_grid, self.y_grid, self.z_grid), self.bx
        )
        self.interpolator_by = RegularGridInterpolator(
            (self.x_grid, self.y_grid, self.z_grid), self.by
        )
        self.interpolator_bz = RegularGridInterpolator(
            (self.x_grid, self.y_grid, self.z_grid), self.bz
        )

    def interpolated_field(self, x, y, z):
        """
        Get interpolated magnetic field at position (x, y, z).

        Uses trilinear interpolation between grid points. Returns zero
        field for points outside the grid bounds.

        Parameters
        ----------
        x : float
            X-coordinate in cm.
        y : float
            Y-coordinate in cm.
        z : float
            Z-coordinate in cm.

        Returns
        -------
        numpy.ndarray
            Field vector [Bx, By, Bz] of shape (3,).
        """
        point = np.array([x, y, z])

        # Bounds checking
        if (point[0] < self.x_grid[0] or point[0] > self.x_grid[-1] or
            point[1] < self.y_grid[0] or point[1] > self.y_grid[-1] or
            point[2] < self.z_grid[0] or point[2] > self.z_grid[-1]):
            return np.array([0.0, 0.0, 0.0]).reshape(3,)

        bx = self.interpolator_bx(point)
        by = self.interpolator_by(point)
        bz = self.interpolator_bz(point)
        return np.array([bx, by, bz]).reshape(3,)

    def find_gradient(self, x, y, z):
        """
        Compute field gradient using finite differences.

        Parameters
        ----------
        x, y, z : float
            Position coordinates in cm.

        Returns
        -------
        numpy.ndarray
            Gradient vector [dBx/dx, dBy/dy, dBz/dz] of shape (3,).
        """
        point = np.array([x, y, z])

        if (point[0] < self.x_grid[0] or point[0] > self.x_grid[-1] or
            point[1] < self.y_grid[0] or point[1] > self.y_grid[-1] or
            point[2] < self.z_grid[0] or point[2] > self.z_grid[-1]):
            return np.array([0.0, 0.0, 0.0]).reshape(3,)

        delta = 1e-6
        bx_dx = (self.interpolator_bx(point + np.array([delta, 0, 0])) -
                 self.interpolator_bx(point - np.array([delta, 0, 0]))) / (2 * delta)
        by_dy = (self.interpolator_by(point + np.array([0, delta, 0])) -
                 self.interpolator_by(point - np.array([0, delta, 0]))) / (2 * delta)
        bz_dz = (self.interpolator_bz(point + np.array([0, 0, delta])) -
                 self.interpolator_bz(point - np.array([0, 0, delta]))) / (2 * delta)
        return np.array([bx_dx, by_dy, bz_dz]).reshape(3,)

    def plot_interpolated_field(self, x, y, z_values):
        """
        Plot field components along z-axis at fixed (x, y).

        Parameters
        ----------
        x : float
            Fixed x-coordinate.
        y : float
            Fixed y-coordinate.
        z_values : array-like
            Array of z-values to plot.
        """
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

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    # Assuming field data is loaded correctly and processed as per your logic
    field = np.loadtxt('Data/Bfield.rtf')

    # Normalize x, y, z values
    for i in [0, 1, 2]:
        field[:, i] += np.abs(np.min(field[:, i]))
        field[:, i] /= 10

    # Prepare the 4D Bfield array
    B = np.zeros((41, 41, 146, 3))  # Assuming dimensions are correct based on field data
    for r in field:
        B[int(r[0]), int(r[1]), int(r[2])] = r[3:]

    def plot_field_slice(Bfield, z, quiver_plot, magnitude_plot):
        """
        Update the plot for a given z-coordinate with field strength as colour
        and field direction as arrows.
        """
        # Extract Bx, By, Bz for the current z-slice
        Bx = Bfield[:, :, z, 0]
        By = Bfield[:, :, z, 1]
        Bz = Bfield[:, :, z, 2]

        # Update quiver plot
        quiver_plot.set_UVC(Bx, By)

        # Update magnitude plot
        B_magnitude = np.sqrt(Bx**2 + By**2 + Bz**2)
        magnitude_plot.set_array(B_magnitude.ravel())
        # magnitude_plot.autoscale()  # Rescale to the maximum value in the current slice
        # magnitude_plot.changed()  # Mark the magnitude plot as changed

        # Update quiver plot
        quiver_plot.set_UVC(Bx, By)

    def update(val):
        z = int(slider.val)
        plot_field_slice(B, z, quiver, magnitude)
        plt.draw()

    # Create figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Create a meshgrid for x and y coordinates (assuming uniform spacing)
    x = np.arange(B.shape[0])
    y = np.arange(B.shape[1])
    X, Y = np.meshgrid(x, y)

    # Initial plot for z = 0
    initial_z = 0
    Bx = B[:, :, initial_z, 0]
    By = B[:, :, initial_z, 1]
    Bz = B[:, :, initial_z, 2]

    # Initial magnitude plot (colour map)
    B_magnitude = np.sqrt(Bx**2 + By**2 + Bz**2)
    magnitude = ax.contourf(X, Y, B_magnitude, cmap='plasma', levels=50)
    plt.colorbar(magnitude, ax=ax, label='Field Strength |B|')

    # Initial quiver plot (direction arrows)
    quiver = ax.quiver(X, Y, Bx, By, scale=100, scale_units='xy', color='white', alpha = 0)

    # Add a slider for selecting the z-coordinate
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Z', 0, B.shape[2] - 1, valinit=initial_z, valstep=1)

    # Call update function when the slider is changed
    slider.on_changed(update)

    # Show the plot with the slider
    plt.show()
    ##########
    # field_view.py
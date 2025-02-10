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
        # bx, by, bz = bx / 2000 , by / 2000, bz / 2000
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
        # print(f'interpolated filed position {x,y,z}')
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
    
    def find_gradient(self, x, y, z):
        """Find the gradient of the magnetic field at (x, y, z) using RegularGridInterpolator."""
        point = np.array([x, y, z])
        # Check if the point is within bounds
        if (point[0] < self.x_grid[0] or point[0] > self.x_grid[-1] or
            point[1] < self.y_grid[0] or point[1] > self.y_grid[-1] or
            point[2] < self.z_grid[0] or point[2] > self.z_grid[-1]):
            return np.array([0.0, 0.0, 0.0]).reshape(3,)

        # Calculate the gradient using finite differences
        delta = 1e-6
        bx_dx = (self.interpolator_bx(point + np.array([delta, 0, 0])) - self.interpolator_bx(point - np.array([delta, 0, 0])))/(2*delta)
        by_dy = (self.interpolator_by(point + np.array([0, delta, 0])) - self.interpolator_by(point - np.array([0, delta, 0])))/(2*delta)
        bz_dz = (self.interpolator_bz(point + np.array([0, 0, delta])) - self.interpolator_bz(point - np.array([0, 0, delta])))/(2*delta)
        return np.array([bx_dx, by_dy, bz_dz]).reshape(3,)

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
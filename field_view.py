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


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider

# # Assuming field data is loaded correctly and processed as per your logic
# field = np.loadtxt('Data/Bfield.rtf')

# # Normalize x, y, z values
# for i in [0, 1, 2]:
#     field[:, i] += np.abs(np.min(field[:, i]))
#     field[:, i] /= 10

# # Prepare the 4D Bfield array
# B = np.zeros((41, 41, 146, 3))  # Assuming dimensions are correct based on field data
# for r in field:
#     B[int(r[0]), int(r[1]), int(r[2])] = r[3:]

# def plot_field_slice(Bfield, z, quiver_plot, magnitude_plot):
#     """
#     Update the plot for a given z-coordinate with field strength as colour
#     and field direction as arrows.
#     """
#     # Extract Bx, By, Bz for the current z-slice
#     Bx = Bfield[:, :, z, 0]
#     By = Bfield[:, :, z, 1]
#     Bz = Bfield[:, :, z, 2]

#     # Calculate the magnitude of the magnetic field
#     B_magnitude = np.sqrt(Bx**2 + By**2 + Bz**2)

#     # Update quiver plot
#     quiver_plot.set_UVC(Bx, By)

#     # Update magnitude plot
#     magnitude_plot.set_array(B_magnitude.ravel())

#     # Set color limits based on the current slice
#     min_val = np.min(B_magnitude)
#     max_val = np.max(B_magnitude)

#     # Set limits for the color mapping
#     magnitude_plot.set_clim(vmin=min_val, vmax=max_val)

#     # Redraw the color map
#     magnitude_plot.changed()

# def update(val):
#     z = int(slider.val)
#     plot_field_slice(B, z, quiver, magnitude)
#     plt.draw()

# # Create figure and axis for plotting
# fig, ax = plt.subplots(figsize=(8, 6))
# plt.subplots_adjust(left=0.1, bottom=0.25)

# # Create a meshgrid for x and y coordinates (assuming uniform spacing)
# x = np.arange(B.shape[0])
# y = np.arange(B.shape[1])
# X, Y = np.meshgrid(x, y)

# # Initial plot for z = 0
# initial_z = 0
# Bx = B[:, :, initial_z, 0]
# By = B[:, :, initial_z, 1]
# Bz = B[:, :, initial_z, 2]

# # Initial magnitude plot (colour map)
# B_magnitude = np.sqrt(Bx**2 + By**2 + Bz**2)
# magnitude = ax.contourf(X, Y, B_magnitude, cmap='plasma', levels=50)
# plt.colorbar(magnitude, ax=ax, label='Field Strength |B|')

# # Initial quiver plot (direction arrows)
# quiver = ax.quiver(X, Y, Bx, By, scale=50, color='white')

# # Add a slider for selecting the z-coordinate
# ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
# slider = Slider(ax_slider, 'Z', 0, B.shape[2] - 1, valinit=initial_z, valstep=1)

# # Call update function when the slider is changed
# slider.on_changed(update)

# # Show the plot with the slider
# plt.show()




##################



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider

# # Assuming field data is loaded correctly and processed as per your logic
# field = np.loadtxt('Data/Bfield.rtf')

# # Normalize x, y, z values
# for i in [0, 1, 2]:
#     field[:, i] += np.abs(np.min(field[:, i]))
#     field[:, i] /= 10

# # Prepare the 4D Bfield array
# B = np.zeros((41, 41, 146, 3))  # Assuming dimensions are correct based on field data
# for r in field:
#     B[int(r[0]), int(r[1]), int(r[2])] = r[3:]

# def plot_field_slice(Bfield, z, quiver_plot, magnitude_plot, text_annotations):
#     """
#     Update the plot for a given z-coordinate with field strength as colour
#     and field direction as arrows.
#     """
#     # Clear previous text annotations
#     for text in text_annotations:
#         text.remove()
#     text_annotations.clear()

#     # Extract Bx, By, Bz for the current z-slice
#     Bx = Bfield[:, :, z, 0]
#     By = Bfield[:, :, z, 1]
#     Bz = Bfield[:, :, z, 2]

#     # Calculate the magnitude of the magnetic field
#     B_magnitude = np.sqrt(Bx**2 + By**2 + Bz**2)

#     # Update quiver plot
#     quiver_plot.set_UVC(Bx, By)

#     # Update magnitude plot
#     magnitude_plot.set_array(B_magnitude.ravel())

#     # Set color limits based on the current slice
#     min_val = np.min(B_magnitude)
#     max_val = np.max(B_magnitude)

#     # Set limits for the color mapping
#     magnitude_plot.set_clim(vmin=min_val, vmax=max_val)

#     # Redraw the color map
#     magnitude_plot.changed()

#     # Annotate each cell with the numerical value of the magnitude
#     for i in range(B_magnitude.shape[0]):
#         for j in range(B_magnitude.shape[1]):
#             # Place the text in the center of the grid cell
#             text = ax.text(i, j, f'{B_magnitude[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)
#             text_annotations.append(text)

# def update(val):
#     z = int(slider.val)
#     plot_field_slice(B, z, quiver, magnitude, text_annotations)
#     plt.draw()

# # Create figure and axis for plotting
# fig, ax = plt.subplots(figsize=(8, 6))
# plt.subplots_adjust(left=0.1, bottom=0.25)

# # Create a meshgrid for x and y coordinates (assuming uniform spacing)
# x = np.arange(B.shape[0])
# y = np.arange(B.shape[1])
# X, Y = np.meshgrid(x, y)

# # Initial plot for z = 0
# initial_z = 0
# Bx = B[:, :, initial_z, 0]
# By = B[:, :, initial_z, 1]
# Bz = B[:, :, initial_z, 2]

# # Initial magnitude plot (colour map)
# B_magnitude = np.sqrt(Bx**2 + By**2 + Bz**2)
# magnitude = ax.contourf(X, Y, B_magnitude, cmap='plasma', levels=50)
# plt.colorbar(magnitude, ax=ax, label='Field Strength |B|')

# # Initial quiver plot (direction arrows)
# quiver = ax.quiver(X, Y, Bx, By, scale=50, color='white')

# # Initialize a list to hold text annotations
# text_annotations = []

# # Add a slider for selecting the z-coordinate
# ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
# slider = Slider(ax_slider, 'Z', 0, B.shape[2] - 1, valinit=initial_z, valstep=1)

# # Call update function when the slider is changed
# slider.on_changed(update)

# # Show the plot with the slider
# plt.show()

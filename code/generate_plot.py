import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# === Load calibration file ===
calib_path = os.path.join("/home/hojune/Desktop", "E90", "data", "calibration", "full_calibration_new.npz")
data = np.load(calib_path)

# === Extract R, T ===
R0, T0 = data['R_0'], data['T_0']
R1, T1 = data['R_1'], data['T_1']
R2, T2 = data['R_2'], data['T_2']

# === Compute camera center in world coordinates ===
def get_camera_center(R, T):
    return -R.T @ T

# === Function to draw XYZ axes at a given origin with rotation ===
def draw_axes(ax, origin, R, length=0.5):
    origin = origin.reshape(3)
    x_axis = R.T @ np.array([1, 0, 0]) * length
    y_axis = R.T @ np.array([0, 1, 0]) * length
    z_axis = R.T @ np.array([0, 0, 1]) * length

    ax.quiver(*origin, *x_axis, color='r', linewidth=2)
    ax.quiver(*origin, *y_axis, color='g', linewidth=2)
    ax.quiver(*origin, *z_axis, color='b', linewidth=2)

# === Compute camera centers ===
C0 = get_camera_center(R0, T0)
C1 = get_camera_center(R1, T1)
C2 = get_camera_center(R2, T2)
W  = np.array([[0], [0], [0]])  # World origin

# === Plot setup ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# === Annotate each point ===
ax.text(*C0.flatten(), 'Cam0', fontsize=8)
ax.text(*C1.flatten(), 'Cam4', fontsize=8)
ax.text(*C2.flatten(), 'Cam2', fontsize=8)
ax.text(0, 0, 0, 'World', fontsize=8)

# === Draw small XYZ axes in top-right corner ===
# Find bounding box of all plotted points
all_points = np.stack([C0.flatten(), C1.flatten(), C2.flatten(), W.flatten()])
max_x, max_y, max_z = np.max(all_points, axis=0)


# === Draw local axes at each camera and world ===
draw_axes(ax, C0, R0)
draw_axes(ax, C1, R1)
draw_axes(ax, C2, R2)
draw_axes(ax, W, np.eye(3))  # World axes

# === Set axis labels and limits ===
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Camera Centers and Axes in 3D Space")
ax.legend()
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

# === Equalize axis scaling: same units for X, Y, Z ===
all_points = np.stack([C0.flatten(), C1.flatten(), C2.flatten(), W.flatten()])
x_min, y_min, z_min = np.min(all_points, axis=0)
x_max, y_max, z_max = np.max(all_points, axis=0)

# Pad equally in all directions
max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
center_x = (x_max + x_min) / 2
center_y = (y_max + y_min) / 2
center_z = (z_max + z_min) / 2

ax.set_xlim(center_x - max_range / 2, center_x + max_range / 2)
ax.set_ylim(center_y - max_range / 2, center_y + max_range / 2)
ax.set_zlim(center_z - max_range / 2, center_z + max_range / 2)

# === Set view angle (elevation, azimuth) ===
ax.view_init(elev=-180, azim=-10)

# === Dynamically place top-right corner axis frame above all camera labels ===
all_points = np.stack([C0.flatten(), C1.flatten(), C2.flatten(), W.flatten()])
max_x, max_y, max_z = np.max(all_points, axis=0)


legend_elements = [
    Line2D([0], [0], color='r', lw=2, label='X axis'),
    Line2D([0], [0], color='g', lw=2, label='Y axis'),
    Line2D([0], [0], color='b', lw=2, label='Z axis')
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

# === Save plot ===
output_path = "/home/hojune/Desktop/camera_centers_plot.png"
plt.savefig(output_path)
print(f"Plot saved to: {output_path}")

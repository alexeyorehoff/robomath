from math import sin, cos

from pyglm import glm
import numpy as np
from matplotlib import pyplot as plt

def get_conversion_mat(x_offset, y_offset, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), x_offset],
        [np.sin(theta), np.cos(theta), y_offset],
        [0, 0, 1]
    ], dtype=np.float64)


def display_points(x_coords, y_coords):
    plt.figure()
    plt.scatter(x_coords, y_coords)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Laser Scan Points")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    scan_data = np.loadtxt("assets/laserscan.dat")
    angle_coords = np.linspace(-np.pi / 2, np.pi / 2, len(scan_data))

    x_coords = scan_data * np.cos(angle_coords)
    y_coords = scan_data * np.sin(angle_coords)

    uniform_points = np.vstack((x_coords, y_coords, np.ones_like(x_coords)))
    robot_conversion = get_conversion_mat(1, 0.5, np.pi / 4)
    scanner_conversion = get_conversion_mat(0.2, 0.0, np.pi)

    global_points = robot_conversion @ scanner_conversion @ uniform_points
    x_global, y_global = global_points[0, :], global_points[1, :]

    display_points(x_global, y_global)

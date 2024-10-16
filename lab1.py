from quaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt
import glm


def draw_coordinate_frame(subplot, transform, dotted: bool = False):
    colored_axes = (
        (glm.vec3(1, 0, 0), "red"),
        (glm.vec3(0, 1, 0), "green"),
        (glm.vec3(0, 0, 1), "blue")
    )

    linestyle = ":" if dotted else "-"  # Set dotted or solid lines

    for axis, color in colored_axes:
        rotated = transform.rotate(axis)
        # Drawing a line from origin (0, 0, 0) to the rotated axis
        subplot.plot([0, rotated.x], [0, rotated.y], [0, rotated.z], color=color, linestyle=linestyle)


def random_axis_angle():
    axis = np.random.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    angle = np.random.uniform(0, 2 * np.pi)
    return axis, angle


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    draw_coordinate_frame(ax, Quaternion())
    axis, angle = random_axis_angle()
    print(f"Rotating coordinate system around {tuple(map(lambda el: round(float(el), 3),  axis))} by {angle:.3f} rad")
    draw_coordinate_frame(ax, Quaternion.from_axis_angle(*random_axis_angle()), dotted=True)

    plt.show()


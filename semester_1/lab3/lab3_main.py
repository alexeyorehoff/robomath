import numpy as np

def lab3_1(r, alpha, d, theta):
    """
    Constructs a DH transformation matrix given the parameters.
    :param r: Link length (a)
    :param alpha: Link twist (alpha)
    :param d: Link offset (d)
    :param theta: Joint angle (theta)
    :return: 4x4 DH transformation matrix
    """
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), r * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), r * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ]).round(3)

def lab3_2(theta_1, theta_2, theta_3, theta_4, theta_5, g):
    """
    Computes the resulting coordinates using DH parameters and transformations.
    :param theta_1: Joint angle for joint 1
    :param theta_2: Joint angle for joint 2
    :param theta_3: Joint angle for joint 3
    :param theta_4: Joint angle for joint 4
    :param theta_5: Joint angle for joint 5
    :param g: Gripper width
    :return: Array of resulting coordinates
    """
    a, b, c, d, e = 3, 5.75, 7.375, 4.125, 1.125

    local_coordinates = np.array([0, 0, 0, 1])
    local_coordinates_grab = np.array([
        [0, 0, -e, 1],
        [g / 2, 0, -e, 1],
        [-g / 2, 0, -e, 1],
        [g / 2, 0, 0, 1],
        [-g / 2, 0, 0, 1]
    ])

    A_10 = lab3_1(0, -np.pi / 2, a, theta_1)
    A_21 = lab3_1(b, 0, 0, theta_2 - np.pi / 2)
    A_32 = lab3_1(c, 0, 0, np.pi / 2 + theta_3)
    A_43 = lab3_1(0, -np.pi / 2, 0, -np.pi / 2 + theta_4)
    A_54 = lab3_1(0, 0, d, theta_5)

    coord = []
    coord.append(local_coordinates)
    coord.append(A_10 @ local_coordinates)
    coord.append(A_10 @ A_21 @ local_coordinates)
    coord.append(A_10 @ A_21 @ A_32 @ local_coordinates)
    coord.append(A_10 @ A_21 @ A_32 @ A_43 @ local_coordinates)

    A_50 = A_10 @ A_21 @ A_32 @ A_43 @ A_54
    for grab_coord in local_coordinates_grab:
        coord.append(A_50 @ grab_coord)

    result_coordinates = np.array(coord)[:, :3]
    return result_coordinates


if __name__ == "__main__":
    print(lab3_1(5, 0, 3, np.pi / 2))
    print(lab3_2(0, 0, 0, 0, 0, 0))
    print(lab3_2(np.pi, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 6, 2))

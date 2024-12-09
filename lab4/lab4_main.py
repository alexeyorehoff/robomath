import numpy as np

def dh_to_mat(r, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), r * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), r * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def RPR_FK(theta_1, d_2, theta_3):
    a = 10
    b = 5
    local_coordinates = np.array([0, 0, 0, 1])

    A_10 = dh_to_mat(0, -3 * np.pi / 4, a, theta_1)
    A_21 = dh_to_mat(0, -np.pi / 2, d_2, -np.pi / 2)
    A_32 = dh_to_mat(0, np.pi / 2, 0, np.pi / 4 + theta_3)
    A_43 = dh_to_mat(0, 0, b, np.pi / 2)

    coord = np.zeros((4, 4))
    coord[0, :] = local_coordinates
    coord[1, :] = (A_10 @ local_coordinates)
    coord[2, :] = (A_10 @ A_21 @ local_coordinates)
    coord[3, :] = (A_10 @ A_21 @ A_32 @ A_43 @ local_coordinates)

    coord_matrix = coord[:, :3]
    A_40 = A_10 @ A_21 @ A_32 @ A_43
    rotation_matrix = A_40[:3, :3]

    return coord_matrix.round(3), rotation_matrix.round(3)

def RPR_IK(x, y, z, R):
    a = 10
    b = 5

    theta_3 = np.arctan2(R[2, 2], -R[2, 1])
    d_2 = b * (np.sin(theta_3 + np.pi / 4) - np.cos(theta_3 + np.pi / 4)) - np.sqrt(2) * z + np.sqrt(2) * a

    sin_theta_1 = -np.sqrt(2) * x / (a * np.sin(theta_3 + np.pi / 4) - np.sqrt(2) * z + np.sqrt(2) * a)
    cos_theta_1 = np.sqrt(2) * y / (a * np.sin(theta_3 + np.pi / 4) - np.sqrt(2) * z + np.sqrt(2) * a)

    eps = 0.001
    if abs(sin_theta_1) > 1 + eps or abs(cos_theta_1) > 1 + eps:
        return None  # Нет решения
    else:
        theta_1 = np.arctan2(sin_theta_1, cos_theta_1)

    return np.array([theta_1, d_2, theta_3])


if __name__ == "__main__":
    fk_res, R033 = RPR_FK(np.pi, -10, np.pi / 2)
    ik_res = RPR_IK(fk_res[3, 0], fk_res[3, 1], fk_res[3, 2], R033)
    fk2_res, R034 = RPR_FK(ik_res[0], ik_res[1], ik_res[2])

    print("Forward Kinematics Result:")
    print(fk_res)
    print("\nRecomputed Forward Kinematics Result:")
    print(fk2_res)

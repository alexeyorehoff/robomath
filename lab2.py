import numpy as np
from quaternion import Quaternion
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glm
import math


task1_input = (
    ((-0.1457, 0.5976, -0.7884), 3.5112),
    ((0.4928, 0.5435, 0.6795), 3.5366),
    ((-0.1784, 0.2396, 0.9543), 1.8534),
    ((-0.5780, -0.7786, -0.2442), 1.2844),
    ((0.7362, 0.0666, 0.6734), 4.0863),
    ((-0.6893, 0.6863, 0.2319), 5.0171),
    ((-0.1380, -0.8528, -0.5037), 6.1800),
    ((0.0351, 0.5640, -0.8251), 2.4076),
    ((0.6360, 0.1757, 0.7515), 2.9780),
    ((0.2821, 0.0936, -0.9548), 1.8078),
    ((0.8807, 0.4069, -0.2426), 1.6467),
    ((-0.4320, 0.3838, 0.8162), 1.8309),
)


task2_input = (
    (-0.4161, 0.3523, -0.3074, 0.7800),
    (0.9010, -0.0131, -0.3935, 0.1818),
    (-0.6497, -0.3817, -0.4074, 0.5159),
    (0.8238, 0.0256, 0.1482, 0.5466),
    (0.4707, 0.6699, 0.5226, 0.2377),
    (0.8826, 0.3873, -0.1206, -0.2376),
    (0.6442, -0.5851, -0.3146, 0.3791),
    (-0.3169, 0.1932, -0.6358, 0.6768),
    (-0.7757, 0.5270, -0.2611, -0.2290),
    (-0.1433, -0.5519, -0.6894, -0.4467),
    (-0.8954, 0.2335, -0.2158, 0.3116),
    (0.4678, -0.6865, 0.2708, 0.4863),
)


task3_input = (
    glm.mat3(-0.5092, -0.0269, 0.8602, 0.7973, 0.3617, 0.4833, -0.3242, 0.9319, -0.1628),
    glm.mat3(-0.4434, -0.4486, 0.7760, 0.8961, -0.2012, 0.3957, -0.0214, 0.8708, 0.4912),
    glm.mat3(-0.1350, -0.1968, 0.9711, 0.7667, -0.6416, -0.0234, 0.6277, 0.7413, 0.2375),
    glm.mat3(0.2233, 0.7083, 0.6697, -0.7767, -0.5402, 0.3107, 0.5818, 0.4544, -0.6746),
    glm.mat3(0.2117, -0.0352, 0.9767, 0.7344, -0.7459, -0.1666, 0.6441, 0.6652, -0.1352),
    glm.mat3(-0.4777, -0.3495, 0.8060, 0.8728, 0.0549, 0.4845,  0.1000, 0.8899, 0.4451),
    glm.mat3(0.0028, -0.0401, 0.9992, 0.8698, -0.4929, -0.0222, 0.4934, 0.8692, 0.0198),
    glm.mat3(-0.8299, 0.2217, 0.5120, 0.3220, -0.5592, 0.7654,  0.4557, 0.7993, 0.3944),
    glm.mat3(-0.1631, 0.9346, 0.3162, -0.9840, 0.1773, -0.0197, -0.0716, 0.3084, -0.9458),
    glm.mat3(0.2696, 0.2931, 0.9173, 0.8224, 0.4411, -0.3594, -0.2317, 0.9463, -0.2336),
    glm.mat3(-0.3565, 0.0457, 0.9332, 0.9174, -0.0469, 0.3959, -0.2073, 0.9701, -0.1267),
    glm.mat3(-0.5114, 0.6895, 0.5130, 0.8520, 0.4865, 0.1934, -0.1121, 0.5303, -0.8404)
)


def task1():
    print("Task 1: Преобразовать ось-угол в кватернион")
    for axis, angle in task1_input:
        print(Quaternion.from_axis_angle(axis, angle))


def task2():
    print("Task 2: Преобразовать кватернион в матрицу поворота")
    for quat in task2_input:
        print(Quaternion(*quat).to_rot_mat(), end="\n\n")


def mat_to_axis_angle(mat: glm.mat3) -> ((), float):
    angle = math.acos((mat[0, 0] + mat[1, 1] + mat[2, 2] - 1) / 2)
    axis = glm.vec3(mat[2, 1] - mat[1, 2], mat[0, 2] - mat[2, 0], mat[1, 0] - mat[0, 1]) / 2 / math.sin(angle)
    return tuple(axis), angle


def task3():
    print("Task 2: Преобразовать матрицы в ось-угол")
    for mat in task3_input:
        print(mat_to_axis_angle(mat))


def random_axis_angle():
    axis = np.random.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    angle = np.random.uniform(0, 2 * np.pi)
    return axis, angle


def task4():
    inter_steps = 10
    axis1, angle1 = random_axis_angle()
    axis2, angle2 = random_axis_angle()

    q1 = Quaternion.from_axis_angle(axis1, angle1)
    q2 = Quaternion.from_axis_angle(axis2, angle2)

    # q1 = Quaternion(1, 0, 0, 0)
    # q2 = Quaternion(0, 0, 1, 1).normalize()

    print(f"Quaternion 1: {q1}")
    print(f"Quaternion 2: {q2}\n")

    vector = glm.vec3(1, 0, 0)
    origin = glm.vec3(0, 0, 0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in range(inter_steps + 1):
        t = i / inter_steps
        interpolated_quat = q1.slerp(q2, t)
        rotated_vector = interpolated_quat.rotate(vector)
        ax.quiver(*origin, *rotated_vector, color='r', label='Rotated Vector')
        print(f"t={t:.2f}: {interpolated_quat}")

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    plt.show()


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
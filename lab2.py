from quaternion import Quaternion


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


def task1():
    print("Task 1: Преобразовать ось-угол в кватернион")
    for axis, angle in task1_input:
        print(Quaternion.from_axis_angle(axis, angle))


def task2():
    print("Task 2: Преобразовать кватернион в матрицу поворота")
    for quat in task2_input:
        print(Quaternion(*quat).to_rot_mat(), end="\n\n")


if __name__ == "__main__":
    task1()
    task2()
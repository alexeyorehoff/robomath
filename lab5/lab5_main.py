from numpy import sin, cos, pi
import numpy as np


def dh2mat(a, alpha, d, theta):
    T = np.array([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
                  [sin(theta), cos(theta) * cos(alpha),  -cos(theta) * sin(alpha), a * sin(theta)],
                  [0,          sin(alpha),               cos(alpha),                            d],
                  [0,                   0,                        0,                            1]], np.float16)
    return T.round(3)


def task_1(O1, O2, O3, O4, O5, O1_v, O2_v, O3_v, O4_v, O5_v):
    a = 3
    b = 5.75
    c = 7.375
    d = 4.125
    e = 1.125

    O_v = np.array([[O1_v, O2_v, O3_v, O4_v, O5_v]])

    T1 = dh2mat(0, -pi / 2, a, O1)
    T2 = dh2mat(b, 0, 0, O2 - pi / 2)
    T3 = dh2mat(c, 0, 0, O3 + pi / 2)
    T4 = dh2mat(0, -pi / 2, 0, O4 - pi / 2)
    T5 = dh2mat(0, 0, d, O5)

    T10 = T1
    T20 = np.matmul(T10, T2)
    T30 = np.matmul(T20, T3)
    T40 = np.matmul(T30, T4)
    T50 = np.matmul(T40, T5)

    Z0 = np.array([0, 0, 1])
    Z1 = T10[0:3, 2]
    Z2 = T20[0:3, 2]
    Z3 = T30[0:3, 2]
    Z4 = T40[0:3, 2]
    Z5 = T50[0:3, 2]

    P0 = np.array([0, 0, 0])
    P1 = T10[0:3, 3]
    P2 = T20[0:3, 3]
    P3 = T30[0:3, 3]
    P4 = T40[0:3, 3]
    P5 = T50[0:3, 3]

    J1_l = np.cross(Z0, (P5-P0))
    J2_l = np.cross(Z1, (P5-P1))
    J3_l = np.cross(Z2, (P5-P2))
    J4_l = np.cross(Z3, (P5-P3))
    J5_l = np.cross(Z4, (P5-P4))

    J1_a = Z0
    J2_a = Z1
    J3_a = Z2
    J4_a = Z3
    J5_a = Z4

    J_l = np.stack([J1_l, J2_l, J3_l, J4_l, J5_l], axis=1)
    J_a = np.stack([J1_a, J2_a, J3_a, J4_a, J5_a], axis=1)

    v = J_l.dot(O_v.T).T
    w = J_a.dot(O_v.T).T

    return v, w


def task_2(O1, O2, O3, O4, O5, O6, O1_v, O2_v, O3_v, O4_v, O5_v, O6_v):

    a = 13.0
    b = 2.5
    c = 8.0
    d = 2.5
    e = 8.0
    f = 2.5

    O_v = np.array([[O1_v, O2_v, O3_v, O4_v, O5_v, O6_v]])

    T1 = dh2mat(0, pi / 2, a, O1)
    T2 = dh2mat(c, 0, -b, O2)
    T3 = dh2mat(0, -pi / 2, -d, O3)
    T4 = dh2mat(0, pi / 2, e, O4)
    T5 = dh2mat(0, -pi / 2, 0, O5)
    T6 = dh2mat(0, 0, f, O6)

    T10 = T1
    T20 = np.matmul(T10, T2)
    T30 = np.matmul(T20, T3)
    T40 = np.matmul(T30, T4)
    T50 = np.matmul(T40, T5)
    T60 = np.matmul(T50, T6)

    Z0 = np.array([0, 0, 1])
    Z1 = T10[0:3, 2]
    Z2 = T20[0:3, 2]
    Z3 = T30[0:3, 2]
    Z4 = T40[0:3, 2]
    Z5 = T50[0:3, 2]
    Z6 = T60[0:3, 2]

    P0 = np.array([0, 0, 0])
    P1 = T10[0:3, 3]
    P2 = T20[0:3, 3]
    P3 = T30[0:3, 3]
    P4 = T40[0:3, 3]
    P5 = T50[0:3, 3]
    P6 = T60[0:3, 3]

    J1_l = np.cross(Z0, (P6-P0))
    J2_l = np.cross(Z1, (P6-P1))
    J3_l = np.cross(Z2, (P6-P2))
    J4_l = np.cross(Z3, (P6-P3))
    J5_l = np.cross(Z4, (P6-P4))
    J6_l = np.cross(Z5, (P6-P5))

    J1_a = Z0
    J2_a = Z1
    J3_a = Z2
    J4_a = Z3
    J5_a = Z4
    J6_a = Z5

    J_l = np.stack([J1_l, J2_l, J3_l, J4_l, J5_l, J6_l], axis=1)
    J_a = np.stack([J1_a, J2_a, J3_a, J4_a, J5_a, J6_a], axis=1)

    v = J_l.dot(O_v.T).T
    w = J_a.dot(O_v.T).T

    return v, w


if __name__ == '__main__':
    t1_j_pos_1 = pi/2
    t1_j_pos_2 = -pi/2
    t1_j_pos_3 = pi/2
    t1_j_pos_4 = pi/3
    t1_j_pos_5 = pi/2

    t1_j_vel_1 = 0.1
    t1_j_vel_2 = 0.3
    t1_j_vel_3 = 0.2
    t1_j_vel_4 = -0.1
    t1_j_vel_5 = 0.6

    print(task_1(t1_j_pos_1, t1_j_pos_2, t1_j_pos_3, t1_j_pos_4, t1_j_pos_5,
                 t1_j_vel_1, t1_j_vel_2, t1_j_vel_3, t1_j_vel_4, t1_j_vel_5))

    t2_j_pos_1 = pi/2
    t2_j_pos_2 = -pi/2
    t2_j_pos_3 = pi/4
    t2_j_pos_4 = -pi/6
    t2_j_pos_5 = pi/8
    t2_j_pos_6 = -pi/3

    t2_j_vel_1 = 0.1
    t2_j_vel_2 = -0.2
    t2_j_vel_3 = 0.3
    t2_j_vel_4 = 0.1
    t2_j_vel_5 = 0.4
    t2_j_vel_6 = -0.6

    print(task_1(t2_j_pos_1, t2_j_pos_2, t2_j_pos_3, t2_j_pos_4, t2_j_pos_5,
                 t2_j_vel_1, t2_j_vel_2, t2_j_vel_3, t2_j_vel_4, t2_j_vel_5))

import numpy as np
import math
import matplotlib.pyplot as plt


def plot_icp(x, p, p0, i, rmse):
    plt.cla()
    plt.scatter(x[0, :], x[1, :], c='k', marker='o', s=50, lw=0)
    plt.scatter(p[0, :], p[1, :], c='r', marker='o', s=50, lw=0)
    plt.scatter(p0[0, :], p0[1, :], c='b', marker='o', s=50, lw=0)
    plt.legend(('x', 'p', 'p0'), loc='lower left')
    plt.plot(np.vstack((x[0, :], p[0, :])), np.vstack((x[1, :], p[1, :])), c='k')
    plt.title("Iteration: " + str(i) + "  RMSE: " + str(rmse))
    plt.axis([-10, 15, -10, 15])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.pause(0.5)
    return


def generate_data():
    # create reference data  
    x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -2, -3, -4, -5]])

    # add noise
    p = x + 0.05 * np.random.normal(0, 1, x.shape)

    # translate
    p[0, :] = p[0, :] + 1
    p[1, :] = p[1, :] + 1

    # rotate
    theta1 = (10.0 / 360) * 2 * np.pi
    theta2 = (110.0 / 360) * 2 * np.pi
    rot1 = np.array([[math.cos(theta1), -math.sin(theta1)],
                     [math.sin(theta1), math.cos(theta1)]])
    rot2 = np.array([[math.cos(theta2), -math.sin(theta2)],
                     [math.sin(theta2), math.cos(theta2)]])

    # sets with known correspondences
    p1 = np.dot(rot1, p)
    p2 = np.dot(rot2, p)

    # sets with unknown correspondences
    p3 = np.random.permutation(p1.T).T
    p4 = np.random.permutation(p2.T).T

    return x, p1, p2, p3, p4

import math
from pyglm import glm
import matplotlib.pyplot
from matplotlib import pyplot as plt
import numpy as np


def test_function(x: float) -> float:
    return math.cos(x) * math.exp(x)

def plot_function() -> None:
    xs = np.linspace(-2 * math.pi, 2 * math.pi, 100)
    ys = list(map(test_function, xs))

    plt.scatter(xs, ys)
    plt.savefig("results/cos_exp.png")

def gen_random(array_size = 100_000) -> None:
    np.random.seed = 228
    norm_array = np.random.normal(5.0, 2.0, array_size)
    uniform_array = np.random.uniform(0, 10, array_size)

    print(f"Normal distribution array - {np.mean(norm_array)} mean, {np.std(norm_array)} std")
    print(f"Uniform distribution array - {np.mean(uniform_array)} mean, {np.std(uniform_array)} std")

    matplotlib.pyplot.hist(norm_array, bins=50)
    plt.savefig("results/normal_distribution.png")
    matplotlib.pyplot.hist(uniform_array, bins=50)
    plt.savefig("results/uniform_distribution.png")


def gen_spiral(size: int) -> list:
    coord_range = range(size)
    spiral_direction = [1, 0]
    def rotate_direction(direction: list) -> list:
        return [-direction[1], direction[0]]
    matrix = [[0 for _ in coord_range] for _ in coord_range]
    matrix_pos = [0, 0]
    for idx in range(1, size ** 2 + 1):
        matrix[matrix_pos[1]][matrix_pos[0]] = idx
        next_pos = [matrix_pos[0] + spiral_direction[0], matrix_pos[1] + spiral_direction[1]]
        if not (next_pos[0] in coord_range and next_pos[1] in coord_range and matrix[next_pos[1]][next_pos[0]] == 0):
            spiral_direction = rotate_direction(spiral_direction)
        matrix_pos = [matrix_pos[0] + spiral_direction[0], matrix_pos[1] + spiral_direction[1]]
    return matrix

def draw_matrix(matrix):
    if not matrix or not matrix[0]:
        print("Empty matrix")
        return
    max_width = max(len(str(element)) for row in matrix for element in row)
    print("┌" + "─" * ((max_width + 3) * len(matrix[0]) - 1) + "┐")
    for row in matrix:
        print("│", end="")
        for element in row:
            print(f" {str(element):^{max_width}} │", end="")
        print()
        if row != matrix[-1]:
            print("├" + ("─" * (max_width + 2) + "┼") * (len(row) - 1) + "─" * (max_width + 2) + "┤")
    print("└" + "─" * ((max_width + 3) * len(matrix[0]) - 1) + "┘")


if __name__ == "__main__":
    plot_function()
    gen_random()
    spiral = gen_spiral(10)
    draw_matrix(spiral)
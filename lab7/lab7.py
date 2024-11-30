from __future__ import annotations

import glm
from fontTools.merge.util import first
from glm import vec2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

class Shape:
    def __init__(self, shape_points: tuple[vec2, ...]):
        self._shape = shape_points

    def _get_axes(self) -> list[vec2]:
        # Возвращает список из векторов параллельных сторонам объекта
        return [edge[1] - edge[0] for edge in zip(self._shape, self._shape[1:])]

    def _project(self, axis: vec2) -> tuple[float, float]:
        # Проецирует точки объекта на вектор
        # Возвращает две крайние точки проекции
        min_proj = max_proj = glm.dot(self._shape[0], axis)
        for point in self._shape[1:]:
            proj = glm.dot(point, axis)
            min_proj = min(min_proj, proj)
            max_proj = max(max_proj, proj)
        return min_proj, max_proj

    @staticmethod
    def _overlap(proj1: tuple[float, float], proj2: tuple[float, float]) -> bool:
        return proj1[1] >= proj2[0] and proj2[1] >= proj1[0]

    def check_collision(self, other: Shape) -> bool:
        axes_self = self._get_axes()
        axes_other = other._get_axes()

        for axis in axes_self + axes_other:
            axis /= glm.length(axis)
            proj1 = self._project(axis)
            proj2 = other._project(axis)
            if not self._overlap(proj1, proj2):
                return False
        return True

    def get_points(self) -> list[tuple[float, float]]:
        return [(p.x, p.y) for p in self._shape]


def plot_shapes(shape1: Shape, shape2: Shape, collision: bool):

    fig, ax = plt.subplots()
    patches = []

    poly1 = Polygon(shape1.get_points(), closed=True, edgecolor='blue', fill=True, alpha=0.5)
    poly2 = Polygon(shape2.get_points(), closed=True, edgecolor='red', fill=True, alpha=0.5)

    patches.append(poly1)
    patches.append(poly2)

    patch_collection = PatchCollection(patches, match_original=True)
    ax.add_collection(patch_collection)

    all_points = shape1.get_points() + shape2.get_points()
    xs, ys = zip(*all_points)
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_aspect('equal', adjustable='datalim')

    text = "Collision Detected!" if collision else "No Collision."
    ax.set_title(text, fontsize=14, color='green' if collision else 'red')

    plt.show()


if __name__ == "__main__":
    shape1 = Shape((
        vec2(2, 0),
        vec2(2, 2),
        vec2(0, 2)
    ))

    shape2 = Shape((
        vec2(1, 1),
        vec2(3, 3),
        vec2(1, 3)
    ))

    shape3 = Shape((
        vec2(3, 0),
        vec2(5, 0),
        vec2(5, 2),
        vec2(3, 2)
    ))

    collision1 = shape1.check_collision(shape2)
    collision2 = shape1.check_collision(shape3)

    plot_shapes(shape1, shape2, collision1)
    plot_shapes(shape1, shape3, collision2)

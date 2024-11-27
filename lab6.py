import copy
import dataclasses

import numpy as np
from numpy.typing import NDArray
from lab6 import *


def load_grid(save_path: str) -> NDArray[bool]:
    with open(save_path, 'r') as f:
        grid_map = [list(map(int, line.split())) for line in f.readlines()]
    return np.array(grid_map, dtype=bool)


@dataclasses.dataclass
class Node:
    pos: pos_type
    distance: int
    neighbours: list[Node]


class MapGraph:
    def __init__(self, grid_map: NDArray[bool]):
        self._grid_map = grid_map
        self._boundaries = grid_map.shape
        self._elements: dict[pos_type: Node] = {}


def dijkstra(grid_map: NDArray[bool], start: pos_type, finish: pos_type) -> (bool, tuple[pos_type, ...]):
    """ Вычисляет путь по карте, заданной в виде массива булов grid_map, где 0 - пустое пространство и 1 - стена
        Возвращает найден ли путь и массив содержащий промежуточные точки """
    graph = MapGraph(grid_map, )
    return True, ()


if __name__ == "__main__":
    start_pos = (0, 5)
    goal_pos = (10, 5)
    grid_map_mat = load_grid("lab6/assets/lab6_path.txt")
    initial_field = GridMap(grid_map_mat, (
        GridObject(name="Start", pos=start_pos, obj_type=ObjType.start),
        GridObject(name="Goal", pos=goal_pos, obj_type=ObjType.finish),
    ))

    has_res, path = dijkstra(grid_map_mat, start_pos, goal_pos)
    if has_res:
        dijkstra_field = copy.deepcopy(initial_field)
        dijkstra_field.display_grid("lab6/results/dijkstra/0.jpg")
        dijkstra_field.place_object(GridObject("Player", start_pos, ObjType.player))
        for num, pos in enumerate(path):
            dijkstra_field.move_object("Player", pos)
            dijkstra_field.display_grid(f"lab6/results/dijkstra/{num + 1}.jpg")




from __future__ import annotations
import copy
import dataclasses
from typing import Optional
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
    distance: Optional[float] = float('inf')  # Infinity for unvisited nodes
    neighbors: list[Node] = dataclasses.field(default_factory=list)


class MapGraph:
    def __init__(self, grid_map: NDArray[bool]):
        self._grid_map = grid_map
        self._boundaries = grid_map.shape
        self._elements: dict[pos_type, Node] = self._create_graph()

    def _is_within_bounds(self, pos: pos_type) -> bool:
        x, y = pos
        return 0 <= x < self._boundaries[0] and 0 <= y < self._boundaries[1]

    def _is_walkable(self, pos: pos_type) -> bool:
        x, y = pos
        return self._grid_map[x, y]

    def _get_neighbors(self, pos: pos_type) -> list[pos_type]:
        x, y = pos
        possible_moves = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        return [neighbor for neighbor in possible_moves if self._is_within_bounds(neighbor) and self._is_walkable(neighbor)]

    def _create_graph(self) -> dict[pos_type, Node]:
        graph = {}
        for x in range(self._boundaries[0]):
            for y in range(self._boundaries[1]):
                if self._grid_map[x, y]:  # Create nodes only for walkable spaces
                    pos = (x, y)
                    graph[pos] = Node(pos=pos)

        for pos, node in graph.items():
            node.neighbors = [graph[neighbor] for neighbor in self._get_neighbors(pos)]

        return graph

    def get_nodes(self) -> set[Node]:
        return set(self._elements.values())

    def get_node(self, pos: pos_type) -> Node:
        return self._elements.get(pos, None)


def dijkstra(grid_map: NDArray[bool], start: pos_type, finish: pos_type) -> (bool, tuple[pos_type, ...]):
    """ Вычисляет путь по карте, заданной в виде массива булов grid_map, где 0 - пустое пространство и 1 - стена
        Возвращает найден ли путь и массив содержащий промежуточные точки """
    graph = MapGraph(grid_map)
    if not (start_node := graph.get_node(start)):
        raise AttributeError(f"Illegal start position value ({start}) doesn't correspond to any cell in grid map")
    if not (finish_node := graph.get_node(finish)):
        raise AttributeError(f"Illegal start position value ({start}) doesn't correspond to any cell in grid map")
    visited_nodes: set[Node] = set()
    unvisited_nodes: set[Node] = graph.get_nodes()
    path_found = False

    while len(unvisited_nodes) > 0 and not path_found:
        pass

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




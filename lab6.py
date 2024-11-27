from __future__ import annotations
import copy
import heapq
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
    cost: float = float("inf")
    neighbors: list[Node] = dataclasses.field(default_factory=list)
    previous: Node | None = None

    def __hash__(self):
        return hash(self.pos)

    def __gt__(self, other: Node):
        return self.cost > other.cost


class MapGraph:
    def __init__(self, grid_map: NDArray[bool]):
        self._grid_map = grid_map
        self._boundaries = grid_map.shape
        self._elements: dict[pos_type, Node] = self._create_graph()

    def _is_within_bounds(self, pos: pos_type) -> bool:
        y, x = pos
        return 0 <= x < self._boundaries[0] and 0 <= y < self._boundaries[1]

    def _is_walkable(self, pos: pos_type) -> bool:
        y, x = pos
        return self._grid_map[x, y] == 0

    def _get_neighbors(self, pos: pos_type) -> list[pos_type]:
        y, x = pos
        possible_moves = [(y + dy, x + dx) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        return [neighbor for neighbor in possible_moves if self._is_within_bounds(neighbor) and self._is_walkable(neighbor)]

    def _create_graph(self) -> dict[pos_type, Node]:
        graph = {}
        for x in range(self._boundaries[1]):
            for y in range(self._boundaries[0]):
                if self._grid_map[x, y] == 0:  # Create nodes only for walkable spaces
                    pos = (y, x)
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
    start_node = graph.get_node(start)
    finish_node = graph.get_node(finish)

    if start_node is None:
        raise AttributeError(f"Start position {start} is not valid.")
    if finish_node is None:
        raise AttributeError(f"Finish position {finish} is not valid.")
    priority_queue: list[Node] = []
    start_node.cost = 0
    heapq.heappush(priority_queue, start_node)
    visited_nodes = set()

    while priority_queue:
        current_node = heapq.heappop(priority_queue)
        if current_node == finish_node:
            path = []
            while current_node:
                path.append(current_node.pos)
                current_node = current_node.previous
            return True, tuple(reversed(path))
        elif current_node in visited_nodes:
            continue
        visited_nodes.add(current_node)
        for neighbour in current_node.neighbors:
            if neighbour in visited_nodes:
                continue
            new_cost = current_node.cost + 1
            if new_cost < neighbour.cost:
                neighbour.cost = new_cost
                neighbour.previous = current_node
                heapq.heappush(priority_queue, neighbour)
    return False, ()


if __name__  == "__main__":
    start_pos = (0, 5)
    goal_pos = (10, 5)
    grid_map_mat = load_grid("lab6/assets/lab6_path.txt")
    initial_field = GridMap(grid_map_mat, (
        GridObject(name="Start", pos=start_pos, obj_type=ObjType.start),
        GridObject(name="Goal", pos=goal_pos, obj_type=ObjType.finish),
    ))

    has_res, found_path = dijkstra(grid_map_mat, start_pos, goal_pos)
    if has_res:
        dijkstra_field = copy.deepcopy(initial_field)
        dijkstra_field.display_grid("lab6/results/dijkstra/0.jpg")
        dijkstra_field.place_object(GridObject("Player", start_pos, ObjType.player))
        for num, pos in enumerate(found_path):
            dijkstra_field.move_object("Player", pos)
            dijkstra_field.display_grid(f"lab6/results/dijkstra/{num + 1}.jpg")




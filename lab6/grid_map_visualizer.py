import dataclasses
import enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray


pos_type = tuple[int, int]


class ObjType(enum.Enum):
    space = 0
    wall = 1
    player = 2
    start = 3
    finish = 4


OBJ_COLORS = np.array(((1, 1, 1),
                       (0, 0, 0),
                       (1, 1, 0),
                       (1, 0, 0),
                       (0, 1, 0)))


@dataclasses.dataclass
class GridObject:
    name: str
    pos: pos_type
    obj_type: ObjType


class GridMap:
    def __init__(self, grid_mat: NDArray[bool], objects: tuple[GridObject, ...] | None = None):
        self._grid: NDArray[bool] = grid_mat
        self._objects = {}
        for obj in objects:
            self.place_object(obj)

    def place_object(self, obj: GridObject) -> None:
        self._validate_position(obj.pos)
        self._objects[obj.name] = obj

    def _validate_position(self, position: tuple[int, int]) -> None:
        if self._grid[position[1], position[0]] == 1:
            raise(ValueError(f"Illegal position for an object {position}"))

    def move_object(self, obj_name: str, new_pos: pos_type):
        self._objects[obj_name].pos = new_pos

    def display_grid(self, save_path: str):
        display_grid = np.full(self._grid.shape, ObjType.space.value, dtype=int)
        display_grid[self._grid] = ObjType.wall.value
        for obj in self._objects.values():
            display_grid[obj.pos[1], obj.pos[0]] = obj.obj_type.value

        cmap = ListedColormap(OBJ_COLORS)

        plt.figure(figsize=(8, 8))
        plt.imshow(display_grid, cmap=cmap, origin='upper')

        num_rows, num_cols = self._grid.shape
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-0.5, num_cols, 1), [])
        plt.yticks(np.arange(-0.5, num_rows, 1), [])

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
import dataclasses
import enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from numpy.typing import NDArray


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
    pos: tuple[int, int]
    obj_type: ObjType


class GridMap:
    def __init__(self, file_path: str, objects: tuple[GridObject, ...] | None = None):
        self._grid: NDArray[bool] = self.load_grid(file_path)
        for obj in objects:
            self._validate_position(obj.pos)
        self._objects = {obj.name: obj for obj in (objects or [])}

    @staticmethod
    def load_grid(save_path: str) -> NDArray[bool]:
        with open(save_path, 'r') as f:
            grid_map = [list(map(int, line.split())) for line in f.readlines()]
        return np.array(grid_map, dtype=bool)

    def _validate_position(self, position: tuple[int, int]) -> None:
        if self._grid[position[1], position[0]] == 1:
            raise(ValueError(f"Illegal position for an object {position}"))

    def move_object(self, obj_name: str, new_pos: tuple[int, int]):
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


if __name__ == "__main__":
    field = GridMap("assets/lab6_path.txt", (
        GridObject(name="Start", pos=(0, 5), obj_type=ObjType.start),
        GridObject(name="Goal", pos=(10, 5), obj_type=ObjType.finish),
        GridObject(name="Player", pos=(0, 5), obj_type=ObjType.player)
    ))

    # Move the player and save the resulting grid
    # field.move_object("Player", (2, 2))
    field.move_object("Player", (1, 5))
    field.display_grid("output_grid.png")
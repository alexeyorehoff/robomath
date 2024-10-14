from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import glm


@dataclass
class Quaternion:
    w: float = 1
    x: float = 0
    y: float = 0
    z: float = 0

    @classmethod
    def from_axis_angle(cls, axis, angle) -> Quaternion:
        # axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2.0
        sin_half_angle = np.sin(half_angle)

        w = np.cos(half_angle)
        x = axis[0] * sin_half_angle
        y = axis[1] * sin_half_angle
        z = axis[2] * sin_half_angle

        return cls(w, x, y, z)

    def __mul__(self, q: Quaternion) -> Quaternion:
        return Quaternion(
            self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z,
            self.w * q.x + self.x * q.w + self.y * q.z - self.z * q.y,
            self.w * q.y - self.x * q.z + self.y * q.w + self.z * q.x,
            self.w * q.z + self.x * q.y - self.y * q.x + self.z * q.w
        )

    def conjugate(self) -> Quaternion:
        return Quaternion(self.w, -self.x, -self.y, -self.z)


if __name__ == "__main__":
    random_axis1 = np.random.rand(3)
    random_angle1 = np.random.uniform(0, np.pi * 2)
    random_axis2 = np.random.rand(3)
    random_angle2 = np.random.uniform(0, np.pi * 2)

    glm_quat1 = glm.quat(glm.angleAxis(random_angle1, glm.vec3(random_axis1)))
    my_quat1 = Quaternion.from_axis_angle(random_axis1, random_angle1)
    glm_quat2 = glm.quat(glm.angleAxis(random_angle2, glm.vec3(random_axis2)))
    my_quat2 = Quaternion.from_axis_angle(random_axis2, random_angle2)

    my_res = my_quat1 * my_quat2
    glm_res = glm_quat1 * glm_quat2

    print("Результат умножения кватернионов pyglm: ", glm_res)
    print("Результат самописного умножения кватернионов: ", my_res)
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

    def to_axis_angle(self) -> (tuple, float):
        sin_half_angle = 1 - self.w ** 2
        x = self.x / sin_half_angle
        y = self.y / sin_half_angle
        z = self.z / sin_half_angle
        return (x, y, z), 2 * np.acos(self.w)


    def to_rot_mat(self) -> glm.mat3:
        return glm.mat3(
            1 - 2 * self.y ** 2 - 2 * self.z ** 2, 2 * self.x * self.y - 2 * self.z * self.w, 2 * self.x * self.z + 2 * self.y * self.w,
            2 * self.x * self.y + 2 * self.z * self.w, 1 - 2 * self.x ** 2 - 2 * self.z ** 2, 2 * self.y * self.z - 2 * self.x * self.w,
            2 * self.x * self.z - 2 * self.y * self.w, 2 * self.y * self.w + 2 * self.x * self.w, 1 - 2 * self.x ** 2 - 2 * self.y ** 2
        )

    def __mul__(self, q: Quaternion) -> Quaternion:
        return Quaternion(
            self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z,
            self.w * q.x + self.x * q.w + self.y * q.z - self.z * q.y,
            self.w * q.y - self.x * q.z + self.y * q.w + self.z * q.x,
            self.w * q.z + self.x * q.y - self.y * q.x + self.z * q.w
        )

    def conjugate(self) -> Quaternion:
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __repr__(self):
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"

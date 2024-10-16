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
        return (x, y, z), 2 * np.arccos(self.w)

    def to_rot_mat(self) -> glm.mat3:
        return glm.mat3(
            1 - 2 * self.y ** 2 - 2 * self.z ** 2, 2 * self.x * self.y - 2 * self.z * self.w, 2 * self.x * self.z + 2 * self.y * self.w,
            2 * self.x * self.y + 2 * self.z * self.w, 1 - 2 * self.x ** 2 - 2 * self.z ** 2, 2 * self.y * self.z - 2 * self.x * self.w,
            2 * self.x * self.z - 2 * self.y * self.w, 2 * self.y * self.w + 2 * self.x * self.w, 1 - 2 * self.x ** 2 - 2 * self.y ** 2
        )

    def __add__(self, quat: Quaternion) -> Quaternion:
        return Quaternion(self.w + quat.w, self.x + quat.x, self.y + quat.y, self.z + quat.z)


    def __sub__(self, quat: Quaternion) -> Quaternion:
        return Quaternion(self.w - quat.w, self.x - quat.x, self.y - quat.y, self.z - quat.z)


    def __mul__(self, other: Quaternion | float) -> Quaternion | float:
        if isinstance(other, Quaternion):
            return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
        return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other: float) -> Quaternion:
        return Quaternion(self.w / other, self.x / other, self.y / other, self.z / other)

    def __neg__(self) -> Quaternion:
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __matmul__(self, q: Quaternion) -> Quaternion:
        return Quaternion(
            self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z,
            self.w * q.x + self.x * q.w + self.y * q.z - self.z * q.y,
            self.w * q.y - self.x * q.z + self.y * q.w + self.z * q.x,
            self.w * q.z + self.x * q.y - self.y * q.x + self.z * q.w
        )

    def rotate(self, vector: glm.vec3) -> glm.vec3:
        vector_quat = Quaternion(0, *vector)
        rotated_quat = self @ vector_quat @ self.inv
        return glm.vec3(rotated_quat.x, rotated_quat.y, rotated_quat.z)

    @property
    def conj(self) -> Quaternion:
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    @property
    def norm(self) -> float:
        return np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    @property
    def inv(self) -> Quaternion:
        return self.conj * (1 / (self.norm ** 2))

    def normalize(self) -> Quaternion:
        norm = self.norm
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)

    def __repr__(self):
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"

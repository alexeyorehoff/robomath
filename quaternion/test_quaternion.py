import unittest
import numpy as np
import glm
from quaternion import Quaternion

np.random.seed(228)


class TestQuaternion(unittest.TestCase):
    num_cases = 5

    def test_from_axis_angle(self):
        for _ in range(self.num_cases):
            axis = np.random.rand(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, np.pi * 2)

            glm_quat = glm.quat(glm.angleAxis(angle, glm.vec3(axis)))
            my_quat = Quaternion.from_axis_angle(axis, angle)

            self.assertAlmostEqual(glm_quat.w, my_quat.w, 6)
            self.assertAlmostEqual(glm_quat.x, my_quat.x, 6)
            self.assertAlmostEqual(glm_quat.y, my_quat.y, 6)
            self.assertAlmostEqual(glm_quat.z, my_quat.z, 6)

    def test_to_axis_angle(self):
        for _ in range(self.num_cases):
            axis = np.random.rand(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, np.pi * 2)

            quat = Quaternion.from_axis_angle(axis, angle)
            res_axis, res_angle = quat.to_axis_angle()
            res_axis /= np.linalg.norm(res_axis)

            self.assertAlmostEqual(angle, res_angle, 6)
            self.assertAlmostEqual(res_axis[0], axis[0], 6)
            self.assertAlmostEqual(res_axis[1], axis[1], 6)
            self.assertAlmostEqual(res_axis[2], axis[2], 6)

    def test_multiplication(self):
        for _ in range(self.num_cases):
            quat1 = np.random.rand(4)
            quat2 = np.random.rand(4)

            glm_quat1 = glm.quat(*quat1)
            glm_quat2 = glm.quat(*quat2)
            my_quat1 = Quaternion(*quat1)
            my_quat2 = Quaternion(*quat2)

            glm_res = glm_quat1 * glm_quat2
            my_res = my_quat1 @ my_quat2

            self.assertAlmostEqual(glm_res.w, my_res.w, 6)
            self.assertAlmostEqual(glm_res.x, my_res.x, 6)
            self.assertAlmostEqual(glm_res.y, my_res.y, 6)
            self.assertAlmostEqual(glm_res.z, my_res.z, 6)

from typing import List

import taichi as ti
import taichi.math as tm
import numpy as np

from .geometry import Geometry
from .materials import MaterialLibrary, Material


@ti.data_oriented
class UniformSampler:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    # Generates a uniformly-sampled ray direction on the sphere using two canonical random variables
    def sample_direction() -> tm.vec3:

        # Algorithm as seen in lecture slides
        xi_1 = ti.random()
        xi_2 = ti.random()

        w_z = 2.0 * xi_1 - 1.0
        r = tm.sqrt(1.0 - tm.pow(w_z, 2))

        theta = 2.0 * tm.pi * xi_2
        w_x = r * tm.cos(theta)
        w_y = r * tm.sin(theta)

        return tm.vec3(w_x, w_y, w_z)

    @staticmethod
    @ti.func
    def evaluate_probability() -> float:
        return 1. / (4. * tm.pi)


@ti.data_oriented
class BRDF:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    def sample_direction(material: Material, w_o: tm.vec3, normal: tm.vec3) -> tm.vec3:

        # Sampling routine for canonical orientation
        alpha = material.Ns
        xi_1 = ti.random()
        xi_2 = ti.random()

        w_z = tm.pow(xi_1, 1.0 / (alpha + 1.0))
        r = tm.sqrt(1.0 - tm.pow(w_z, 2))
        theta = 2.0 * tm.pi * xi_2
        w_x = r * tm.cos(theta)
        w_y = r * tm.sin(theta)

        w = tm.vec3(w_x, w_y, w_z)

        ortho = tm.mat3(0.0)

        # Rotate direction into appropriate coordinate system
        if alpha == 1:  # diffuse

            # Orthonormal basis aligned with normal
            ortho = ortho_frames(normal)

        elif alpha > 1:  # phong

            # Orthonormal basis aligned with reflected viewing direction
            ortho = ortho_frames(reflect(w_o, normal))

        # Rotate direction using orthonormal frame (brings vector from local to world space)
        w = ortho @ w

        return w

    @staticmethod
    @ti.func
    def evaluate_probability(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> float:

        alpha = material.Ns
        pdf = 0.0

        if alpha == 1:  # Diffuse
            pdf = (1.0 / tm.pi) * tm.max(0, tm.dot(normal, w_i))

        elif alpha > 1:  # Phong
            w_r = reflect(w_o, normal)
            pdf = ((alpha + 1.0) / (2 * tm.pi)) * tm.max(0, tm.pow(tm.dot(w_r, w_i), alpha))

        return pdf

    @staticmethod
    @ti.func
    def evaluate_brdf(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:

        alpha = material.Ns
        p_d = material.Kd

        f_r = tm.vec3(0)

        if alpha == 1:  # Diffuse

            f_r = p_d / tm.pi

        elif alpha > 1:  # Phong
            w_r = reflect(w_o, normal)  # reflected view-direction
            f_r = (p_d * (alpha + 1.0)) / (2.0 * tm.pi) * tm.max(0, (tm.pow(tm.dot(w_r, w_i), alpha)))

        return f_r

    # Computes BRDF factor to resolve instability
    @staticmethod
    @ti.func
    def evaluate_brdf_factor(material: Material, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:

        alpha = material.Ns
        p_d = material.Kd
        p_s = material.Kd

        factor = tm.vec3(0)

        # Simplification given in tutorials
        if alpha == 1:
            factor = p_d

        elif alpha > 1:
            factor = p_s * tm.max(tm.dot(normal, w_i), 0)

        return factor


# Microfacet BRDF based on PBR 4th edition
# https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#
# 546 only deliverable
@ti.data_oriented
class MicrofacetBRDF:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    def sample_direction(material: Material, w_o: tm.vec3, normal: tm.vec3) -> tm.vec3:
        pass

    @staticmethod
    @ti.func
    def evaluate_probability(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> float:
        pass

    @staticmethod
    @ti.func
    def evaluate_brdf(material: Material, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        pass


'''
Ignore for now
'''


@ti.data_oriented
class MeshLightSampler:

    def __init__(self, geometry: Geometry, material_library: MaterialLibrary):
        self.geometry = geometry
        self.material_library = material_library

        pass

    def get_emissive_triangle_indices(self) -> List[int]:
        pass

    @ti.kernel
    def compute_emissive_triangle_areas(self):
        pass

    @ti.func
    def compute_triangle_area_given_id(self, triangle_id: int) -> float:
        pass

    @ti.func
    def compute_triangle_area(self, v0: tm.vec3, v1: tm.vec3, v2: tm.vec3) -> float:
        pass

    @ti.kernel
    def compute_cdf(self):
        pass

    @ti.func
    def sample_emissive_triangle(self) -> int:
        pass

    @ti.func
    def evaluate_probability(self) -> float:
        pass

    @ti.func
    def sample_mesh_lights(self, hit_point: tm.vec3):
        pass


# Creates an orthonormal frame along axis of alignment
@ti.func
def ortho_frames(v_z: tm.vec3) -> tm.mat3:

    random_vec = tm.normalize(tm.vec3([ti.random(), ti.random(), ti.random()]))

    x_axis = tm.cross(v_z, random_vec)
    x_axis = tm.normalize(x_axis)

    y_axis = tm.cross(x_axis, v_z)
    y_axis = tm.normalize(y_axis)

    z_axis = tm.normalize(v_z)

    frame = tm.mat3([x_axis, y_axis, v_z]).transpose()

    return frame


# Reflects ray direction along normal
@ti.func
def reflect(ray_direction: tm.vec3, normal: tm.vec3) -> tm.vec3:
    return 2.0 * tm.dot(normal, ray_direction) * normal - ray_direction

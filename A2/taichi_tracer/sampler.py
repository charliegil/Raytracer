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


# TODO: Implement BRDF Sampling Methods
@ti.data_oriented
class BRDF:
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
    def evaluate_brdf(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        pass

    @staticmethod
    @ti.func
    def evaluate_brdf_factor(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        pass


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
    def evaluate_brdf(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
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


@ti.func
def ortho_frames(v_z: tm.vec3) -> tm.mat3:
    pass


@ti.func
def reflect(ray_direction: tm.vec3, normal: tm.vec3) -> tm.vec3:
    pass
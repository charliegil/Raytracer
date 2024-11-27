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
    def evaluate_brdf(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        pass


@ti.data_oriented
class MeshLightSampler:

    def __init__(self, geometry: Geometry, material_library: MaterialLibrary):
        self.geometry = geometry
        self.material_library = material_library

        # Find all of the emissive triangles
        emissive_triangle_ids = self.get_emissive_triangle_indices()
        if len(emissive_triangle_ids) == 0:
            self.has_emissive_triangles = False
        else:
            self.has_emissive_triangles = True
            self.n_emissive_triangles = len(emissive_triangle_ids)
            emissive_triangle_ids = np.array(emissive_triangle_ids, dtype=np.int32)
            self.emissive_triangle_ids = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=ti.int32)
            self.emissive_triangle_ids.from_numpy(emissive_triangle_ids)

        # Setup for importance sampling
        if self.has_emissive_triangles:
            # Data Fields
            self.emissive_triangle_areas = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=float)
            self.cdf = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=float)
            self.total_emissive_area = ti.field(shape=(), dtype=float)

            # Compute
            self.compute_emissive_triangle_areas()
            self.compute_cdf()


    def get_emissive_triangle_indices(self) -> List[int]:
        # Iterate over each triangle, and check for emissivity 
        emissive_triangle_ids = []
        for triangle_id in range(1, self.geometry.n_triangles + 1):
            material_id = self.geometry.triangle_material_ids[triangle_id-1]
            emissivity = self.material_library.materials[material_id].Ke
            if emissivity.norm() > 0:
                emissive_triangle_ids.append(triangle_id)

        return emissive_triangle_ids

    @ti.kernel
    def compute_emissive_triangle_areas(self):
        for i in range(self.n_emissive_triangles):
            triangle_id = self.emissive_triangle_ids[i]
            vert_ids = self.geometry.triangle_vertex_ids[triangle_id-1] - 1  # Vertices are indexed from 1
            v0 = self.geometry.vertices[vert_ids[0]]
            v1 = self.geometry.vertices[vert_ids[1]]
            v2 = self.geometry.vertices[vert_ids[2]]

            triangle_area = self.compute_triangle_area(v0, v1, v2)
            self.emissive_triangle_areas[i] = triangle_area
            self.total_emissive_area[None] += triangle_area

    @ti.func
    def compute_triangle_area(self, v0: tm.vec3, v1: tm.vec3, v2: tm.vec3) -> float:
        ab = v1 - v0
        ac = v2 - v0

        return 0.5 * (tm.cross(ab, ac).norm())

    @ti.kernel
    def compute_cdf(self):

        cdf_sum = 0.0
        ti.loop_config(serialize=True)
        for i in range(self.n_emissive_triangles):
            cdf_sum += self.emissive_triangle_areas[i]
            self.cdf[i] = cdf_sum / self.total_emissive_area[None]

    @ti.func
    def sample_emissive_triangle(self) -> int:
        xi_triangle = ti.random()  # get random variable [0, 1]

        # Binary search over cdf values
        left = 0
        right = self.n_emissive_triangles - 1
        while left < right:
            mid = (left + right) // 2
            if self.cdf[mid] >= xi_triangle:
                right = mid
            else:
                left = mid + 1

        return left

    @ti.func
    def evaluate_probability(self) -> float:
        return 1.0 / self.total_emissive_area[None]

    @ti.func
    def sample_mesh_lights(self, hit_point: tm.vec3):
        sampled_light_triangle_idx = self.sample_emissive_triangle()
        sampled_light_triangle = self.emissive_triangle_ids[sampled_light_triangle_idx]

        # Grab Vertices
        vert_ids = self.geometry.triangle_vertex_ids[sampled_light_triangle-1] - 1  # Vertices are indexed from 1
        
        v0 = self.geometry.vertices[vert_ids[0]]
        v1 = self.geometry.vertices[vert_ids[1]]
        v2 = self.geometry.vertices[vert_ids[2]]

        # generate point on triangle using random barycentric coordinates
        # https://www.pbr-book.org/4ed/Shapes/Triangle_Meshes#Sampling
        # https://www.pbr-book.org/4ed/Shapes/Triangle_Meshes#SampleUniformTriangle

        # given your sampled triangle vertices
        # generate random barycentric coordinates
        xi_0 = ti.random()
        xi_1 = ti.random()

        b0 = 0.0
        b1 = 0.0

        if xi_0 < xi_1:
            b0 = xi_0 / 2.0
            b1 = xi_1 - b0
        else:
            b1 = xi_1 / 2.0
            b0 = xi_0 - b1

        b2 = 1.0 - b0 - b1

        # calculate the light direction (normalized)
        sampled_point = b0 * v0 + b1 * v1 + b2 * v2
        light_direction = tm.normalize(sampled_point - hit_point)

        return light_direction, sampled_light_triangle


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


@ti.func
def reflect(ray_direction: tm.vec3, normal: tm.vec3) -> tm.vec3:
    return 2.0 * tm.dot(normal, ray_direction) * normal - ray_direction
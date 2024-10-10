from enum import IntEnum

import taichi as ti
import taichi.math as tm

from .camera import Camera
from .ray import Ray, HitData
from .scene_data import SceneData

from .sampler import UniformSampler, BRDF, reflect
from .ray import Ray


@ti.data_oriented
class A1Renderer:

    # Enumerate the different shading modes
    class ShadeMode(IntEnum):
        HIT = 1
        TRIANGLE_ID = 2
        DISTANCE = 3
        BARYCENTRIC = 4
        NORMAL = 5
        MATERIAL_ID = 6

    def __init__(
            self,
            width: int,
            height: int,
            scene_data: SceneData
    ) -> None:

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.scene_data = scene_data
        self.iter_counter = ti.field(dtype=float, shape=())

        self.shade_mode = ti.field(shape=(), dtype=int)
        self.set_shade_hit()

        # Distance at which the distance shader saturates
        self.max_distance = 10.

        # Numbers used to generate colors for integer index values
        self.r = 3.14159265
        self.b = 2.71828182
        self.g = 6.62607015

    def set_shade_hit(self):
        self.shade_mode[None] = self.ShadeMode.HIT

    def set_shade_triangle_ID(self):
        self.shade_mode[None] = self.ShadeMode.TRIANGLE_ID

    def set_shade_distance(self):
        self.shade_mode[None] = self.ShadeMode.DISTANCE

    def set_shade_barycentrics(self):
        self.shade_mode[None] = self.ShadeMode.BARYCENTRIC

    def set_shade_normal(self):
        self.shade_mode[None] = self.ShadeMode.NORMAL

    def set_shade_material_ID(self):
        self.shade_mode[None] = self.ShadeMode.MATERIAL_ID

    @ti.kernel
    def render(self):
        self.iter_counter[None] += 1
        for x, y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x, y, True)
            color = self.shade_ray(primary_ray)
            self.canvas[x, y] += (color - self.canvas[x, y]) / self.iter_counter[None]

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)

    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        hit_data = self.scene_data.ray_intersector.query_ray(ray)
        color = tm.vec3(0)
        if self.shade_mode[None] == int(self.ShadeMode.HIT):
            color = self.shade_hit(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.TRIANGLE_ID):
            color = self.shade_triangle_id(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.DISTANCE):
            color = self.shade_distance(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.BARYCENTRIC):
            color = self.shade_barycentric(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.NORMAL):
            color = self.shade_normal(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.MATERIAL_ID):
            color = self.shade_material_id(hit_data)
        return color

    @ti.func
    def shade_hit(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            if not hit_data.is_backfacing:
                color = tm.vec3(1)
            else:
                color = tm.vec3([0.5, 0, 0])
        return color

    @ti.func
    def shade_triangle_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            triangle_id = hit_data.triangle_id + 1  # Add 1 so that ID 0 is not black
            r = triangle_id * self.r % 1
            g = triangle_id * self.g % 1
            b = triangle_id * self.b % 1
            color = tm.vec3(r, g, b)
        return color

    @ti.func
    def shade_distance(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            d = tm.clamp(hit_data.distance / self.max_distance, 0, 1)
            color = tm.vec3(d)
        return color

    @ti.func
    def shade_barycentric(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            u = hit_data.barycentric_coords[0]
            v = hit_data.barycentric_coords[1]
            w = 1. - u - v
            color = tm.vec3(u, v, w)
        return color

    @ti.func
    def shade_normal(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            normal = hit_data.normal
            color = (normal + 1.) / 2.  # Scale to range [0,1]
        return color

    @ti.func
    def shade_material_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            material_id = hit_data.material_id + 1  # Add 1 so that ID 0 is not black
            r = material_id * self.r % 1
            g = material_id * self.g % 1
            b = material_id * self.b % 1
            color = tm.vec3(r, g, b)
        return color


@ti.data_oriented
class A2Renderer:
    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        BRDF = 2
        MICROFACET = 3

    def __init__(
            self,
            width: int,
            height: int,
            scene_data: SceneData
    ) -> None:

        self.RAY_OFFSET = 1e-6

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data

        self.sample_mode = ti.field(shape=(), dtype=int)
        self.set_sample_uniform()

    def set_sample_uniform(self):
        self.sample_mode[None] = self.SampleMode.UNIFORM

    def set_sample_brdf(self):
        self.sample_mode[None] = self.SampleMode.BRDF

    def set_sample_microfacet(self):
        self.sample_mode[None] = self.SampleMode.MICROFACET

    @ti.kernel
    def render(self):
        self.iter_counter[None] += 1

        for x, y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x, y, True)
            color = self.shade_ray(primary_ray)
            self.canvas[x, y] += (color - self.canvas[x, y]) / self.iter_counter[None]

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)

    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:

        # Initialize color to black
        color = tm.vec3(0.)

        # Get hit data
        hit_data = self.scene_data.ray_intersector.query_ray(ray)

        if hit_data.is_hit:
            shading_point = ray.origin + ray.direction * hit_data.distance  # find intersection point

            # Initialize shadow ray
            shadow_ray = Ray()
            shadow_ray.origin = shading_point + (self.RAY_OFFSET * hit_data.normal)  # offset initial position of
            # shadow ray by epsilon along normal

            # Get material properties of intersection
            material = self.scene_data.material_library.materials[hit_data.material_id]  # get hit material
            alpha = material.Ns  # specular component
            p_d = material.Kd  # diffuse reflectance

            # Viewing direction
            w_o = -ray.direction

            # Normal
            normal = hit_data.normal

            # Initialize incident ray, pdf
            w_i = tm.vec3(0.0)
            pdf = 0.0

            # Sample incident ray direction depending on sampling type
            if self.sample_mode[None] == int(self.SampleMode.UNIFORM):

                w_i = UniformSampler.sample_direction()
                pdf = UniformSampler.evaluate_probability()

            # TODO: Implement BRDF Sampling
            elif self.sample_mode[None] == int(self.SampleMode.BRDF):
                w_i = BRDF.sample_direction(material, w_o, normal)
                pdf = BRDF.evaluate_probability(material, w_o, w_i, normal)

            # Micro-facet BRDF Sampling
            elif self.sample_mode[None] == int(self.SampleMode.MICROFACET):
                pass

            # Set shadow ray direction to sampled direction
            shadow_ray.direction = w_i

            # Check visibility of shadow ray in sampled direction
            shadow_hit_data = self.scene_data.ray_intersector.query_ray(shadow_ray)

            # If not occluded, compute color
            if not shadow_hit_data.is_hit:

                # Query environment for l_e
                l_e = self.scene_data.environment.query_ray(shadow_ray)

                # Compute BRDF
                f_r = BRDF.evaluate_brdf(material, w_o, w_i, normal)

                # Compute final color
                # color = (l_e * f_r * tm.max(tm.dot(normal, w_i), 0)) / pdf
                color = l_e

        else:
            color = self.scene_data.environment.query_ray(ray)

        return color

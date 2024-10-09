import taichi
import taichi as ti
import taichi.math as tm
import numpy as np

from .ray import Ray


@ti.data_oriented
class Camera:

    def __init__(self, width: int = 128, height: int = 128) -> None:

        # Camera pixel width and height are fixed
        self.width = width
        self.height = height

        # Camera parameters that can be modified are stored as fields
        self.eye = ti.Vector.field(n=3, shape=(), dtype=float)
        self.at = ti.Vector.field(n=3, shape=(), dtype=float)
        self.up = ti.Vector.field(n=3, shape=(), dtype=float)
        self.fov = ti.field(shape=(), dtype=float)

        self.x = ti.Vector.field(n=3, shape=(), dtype=float)
        self.y = ti.Vector.field(n=3, shape=(), dtype=float)
        self.z = ti.Vector.field(n=3, shape=(), dtype=float)

        self.camera_to_world = ti.Matrix.field(n=4, m=4, shape=(), dtype=float)

        # Initialize with some default params
        self.set_camera_parameters(
            eye=tm.vec3([0, 0, 5]),
            at=tm.vec3([0, 0, 0]),
            up=tm.vec3([0, 1, 0]),
            fov=60.
        )

    def set_camera_parameters(
            self,
            eye: tm.vec3 = None,
            at: tm.vec3 = None,
            up: tm.vec3 = None,
            fov: float = None
    ) -> None:

        if eye:
            self.eye[None] = eye
        if at:
            self.at[None] = at
        if up:
            self.up[None] = up
        if fov:
            self.fov[None] = fov
        self.compute_matrix()

    @ti.kernel
    def compute_matrix(self):

        # Define camera space coordinate frame
        z_c = tm.vec3(self.at[None] - self.eye[None]).normalized()  # vector from ray origin to lookat point, normalized
        x_c = tm.vec3(tm.cross(self.up[None], z_c)).normalized()  # up cross z
        y_c = tm.vec3(tm.cross(z_c, x_c)).normalized()  # x cross z

        # Update coordinate frame
        self.x[None] = x_c
        self.y[None] = y_c
        self.z[None] = z_c

        # Define camera to world transformation matrix
        self.camera_to_world[None] = tm.mat4(
            [x_c.x, y_c.x, z_c.x, self.eye[None].x],
            [x_c.y, y_c.y, z_c.y, self.eye[None].y],
            [x_c.z, y_c.z, z_c.z, self.eye[None].z],
            [0, 0, 0, 1]
        )

    @ti.func
    def generate_ray(self, pixel_x: int, pixel_y: int, jitter: bool) -> Ray:

        # Generate ndc coords from pixel index
        ndc_coords = self.generate_ndc_coords(pixel_x, pixel_y)

        # Apply offset if jitter = True
        if jitter:
            ndc_coords.x += (ti.random() - 0.5) * 2.0 / self.width  # [-0.5, 0.5] * pixel width * intensity (trial and error)
            ndc_coords.y += (ti.random() - 0.5) * 2.0 / self.height  # [-0.5, 0.5] * pixel height

        # Generate camera coordinates from NDC coords
        cam_coords = self.generate_camera_coords(ndc_coords)

        # Compute ray direction
        dir_w = (self.camera_to_world[None] @ cam_coords).xyz.normalized()

        # Initialize ray
        ray = Ray()
        ray.origin = self.eye[None]
        ray.direction = dir_w

        return ray

    #  Generates ndc coordinates from given pixel index, shifts to middle of pixel
    @ti.func
    def generate_ndc_coords(self, pixel_x: int, pixel_y: int, jitter: bool = False) -> tm.vec2:

        ndc_x = (pixel_x + 0.5) / self.width * 2.0 - 1.0
        ndc_y = (pixel_y + 0.5) / self.height * 2.0 - 1.0
        return tm.vec2([ndc_x, ndc_y])

    # NDC to camera space transformation
    @ti.func
    def generate_camera_coords(self, ndc_coords: tm.vec2) -> tm.vec4:

        fov_rad = tm.radians(self.fov[None])  # convert to radians to use with taichi trig functions
        aspect_ratio = self.width / self.height

        cam_x = ndc_coords.x * aspect_ratio * tm.tan(fov_rad / 2.0)
        cam_y = ndc_coords.y * tm.tan(fov_rad / 2.0)
        cam_z = 1.0

        return tm.vec4([cam_x, cam_y, cam_z, 0.0])

## 1. Triangle Intersections and Normal Interpolation

In this assignment, I implemented a simple ray tracer with triangle intersection capabilities and normal interpolation for smooth shading. The environment supports basic scene navigation using the **WASD** keys and includes multiple shading modes to visualize different aspects of our scene.

### Cornell Box Scene

Below is an example of the Cornell box rendered with my raytracer.

![Screenshot 2024-11-11 140048](https://github.com/user-attachments/assets/dc0fdfd6-62d3-4f8a-9c52-7ff08110cefb)

---

## 2. Progressive Rendering, Direct Illumination, and Sampling

In the second assignment, progressive rendering and ray jittering were added to improve image quality by reducing aliasing. Additionally, we implemented Uniform and BRDF direction sampling along with Probability Density Function (PDF) sampling. This setup and the progressive renderer allow us to achieve smooth rendering results in simple scenes with reasonable performance.

### Progressive Renderer in Action

![Untitled video - Made with Clipchamp (2)](https://github.com/user-attachments/assets/817668e1-394e-4488-8e42-0d54d3a53861)

### Sampling Techniques

Below are images showcasing different sampling techniques:

#### Uniform Sampling
![uniform](https://github.com/user-attachments/assets/9015747e-910c-4150-8234-91b0d54f215c)

#### BRDF Sampling
![brdf](https://github.com/user-attachments/assets/8c89bbbd-fd81-43f4-a860-1d423c7c628b)

---

## 3. Mesh Lights, Light Importance Sampling and MIS

In this assignment, I added support for mesh lights, which allows for soft shadows, light importance sampling, and MIS, which allows us to blend BRDF and importance sampling to obtain better results.
The following images show how MIS allows us to converge towards a correct image much faster.

# BRDF Sampling @ 1spp
![image](https://github.com/user-attachments/assets/a9f5d96f-5f9d-443a-8b50-3ff5004ba489)

# Light Importance Sampling @ 1spp
![image](https://github.com/user-attachments/assets/027a0905-db5f-49fa-b78d-19918596215d)

# MIS: 50% BRDF, 50% LIS @ 1spp
![image](https://github.com/user-attachments/assets/54a80fb0-3e13-414a-82d9-da06b1eeab58)

# Final Image @ 100spp
![image](https://github.com/user-attachments/assets/3d44d0aa-04bd-4207-8c6c-d01c89ac5e0e)

---

## 4. Path tracing

I added implicit and explicit path-tracing with Russian roulette termination in this assignment.
As images are much nicer when using explicit path tracing, I have skipped results for implicit path tracing. Here are the results of explicit path tracing for increasing maximum bounces @ 100spp.

# 1 bounce
![image](https://github.com/user-attachments/assets/b920414f-d39e-438b-b510-6f91e17a6129)

# 2 bounces
![image](https://github.com/user-attachments/assets/d33803dd-0a26-4eb7-b9c8-a6899e2b8589)

# 3 bounces
![image](https://github.com/user-attachments/assets/e916413b-2414-498c-b26c-0c6f39d14434)

# 4 bounces
![image](https://github.com/user-attachments/assets/6547f53c-db34-4954-bafd-cd7d00b4c611)

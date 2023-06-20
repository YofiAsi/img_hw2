import numpy as np
from math import sqrt

class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = position
        self.radius = radius
        self.material_index = material_index

    def intersect(self, ray):
        L = self.position - ray.origin
        t_ca = np.dot(L, ray.direction)
        # If sphere is behind us:
        if t_ca < 0:
            return -1

        r_squared = self.radius * self.radius

        d_squared = np.dot(L, L) - t_ca * t_ca
        # If the ray is outside of sphere:
        if d_squared > r_squared:
            return -1
        t_hc = sqrt(r_squared - d_squared)
        t = t_ca - t_hc
        return t

    def reflect(self, ray, hit_point):
        normal = self.calculate_normal(hit_point)
        reflected_direction = ray.direction - 2 * np.dot(ray.direction, normal) * normal
        return reflected_direction

    def refract(self, ray, hit_point, refractive_index=1.5):
        normal = self.calculate_normal(hit_point)
        cos_theta1 = np.dot(-ray.direction, normal)
        cos_theta2 = np.sqrt(1 - (refractive_index**2) * (1 - cos_theta1**2))

        refracted_direction = refractive_index * ray.direction + (refractive_index * cos_theta1 - cos_theta2) * normal
        return refracted_direction

    def calculate_normal(self, point):
        return (point - self.position) / self.radius
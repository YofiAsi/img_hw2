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
        normal = self.calc_normal(hit_point)
        reflected_direction = ray.direction - 2 * np.dot(ray.direction, normal) * normal
        return reflected_direction

    def calc_normal(self, point):
        normal = point - self.position
        normal = normal / np.linalg.norm(normal)
        return normal
    
    def refract(self, ray, hit_point):
        normal = self.calc_normal(hit_point)
        incident_direction = ray.direction
        refractive_index = 1.1

        cos_i = -np.dot(normal, incident_direction)
        sin_t_squared = refractive_index**2 * (1 - cos_i**2)

        if sin_t_squared > 1.0:
            # Total internal reflection, reflect the ray
            reflected_direction = incident_direction - 2 * np.dot(incident_direction, normal) * normal
            return hit_point, reflected_direction
        else:
            cos_t = sqrt(1 - sin_t_squared)
            refracted_direction = refractive_index * incident_direction + (refractive_index * cos_i - cos_t) * normal
            refracted_origin = hit_point - 2 * self.radius * normal  # Calculate origin inside the sphere
            return refracted_origin, refracted_direction


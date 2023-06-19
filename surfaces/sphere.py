import numpy as np

class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = position
        self.radius = radius
        self.material_index = material_index

    def intersect(self, rays):
        origins = rays.origin
        directions = rays.direction

        oc = origins - self.position
        a = np.sum(directions * directions, axis=1)
        b = 2 * np.sum(oc * directions, axis=1)
        c = np.sum(oc * oc, axis=1) - self.radius * self.radius

        discriminant = b * b - 4 * a * c
        mask = discriminant >= 0

        discriminant = np.where(mask, discriminant, np.inf)

        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)

        t = np.where(mask, np.minimum(t1, t2), np.inf)
        t = np.maximum(t, 0)

        hit_points = np.where(mask[:, np.newaxis], origins + t[:, np.newaxis] * directions, None)

        return hit_points

    def refract(self, direction, intersection_point, refractive_index_ratio=1.5):
        surface_normal = (intersection_point - self.position) / self.radius

        cos_theta_i = -np.dot(direction, surface_normal)

        if cos_theta_i > 0:
            # Ray is exiting the sphere, flip the surface normal and invert the refractive index ratio
            surface_normal = -surface_normal
            refractive_index_ratio = 1 / refractive_index_ratio
        else:
            # Ray is entering the sphere, use the surface normal as is
            surface_normal = surface_normal

        cos_theta_t = np.sqrt(1 - refractive_index_ratio**2 * (1 - cos_theta_i**2))

        if np.isnan(cos_theta_t):
            # Total internal reflection
            return None

        refracted_direction = refractive_index_ratio * direction + (refractive_index_ratio * cos_theta_i - cos_theta_t) * surface_normal

        return refracted_direction

    def reflect(self, direction, intersection_point):
        incident_direction = -direction
        surface_normal = (intersection_point - self.position) / self.radius

        reflected_direction = incident_direction - 2 * np.dot(incident_direction, surface_normal) * surface_normal

        return reflected_direction
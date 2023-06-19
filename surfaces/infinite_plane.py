import numpy as np

class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index

    def intersect(self, rays):
        origins = rays.origin
        directions = rays.direction

        denom = np.sum(self.normal * directions, axis=1)
        t = -(np.sum(self.normal * origins, axis=1) + self.offset) / denom

        mask = denom != 0
        t = np.where(mask, t, np.inf)
        t = np.maximum(t, 0)

        hit_points = np.where(mask[:, np.newaxis], origins + t[:, np.newaxis] * directions, None)

        return hit_points
    
    def refract(self, direction, intersection_point, refractive_index_ratio=1.5):
        cos_theta_i = -np.dot(direction, self.normal)

        if cos_theta_i > 0:
            # Ray is exiting the plane, flip the normal and invert the refractive index ratio
            surface_normal = -self.normal
            refractive_index_ratio = 1 / refractive_index_ratio
        else:
            # Ray is entering the plane, use the normal as is
            surface_normal = self.normal

        cos_theta_t = np.sqrt(1 - refractive_index_ratio**2 * (1 - cos_theta_i**2))

        if np.isnan(cos_theta_t):
            # Total internal reflection
            return None

        refracted_direction = refractive_index_ratio * direction + (refractive_index_ratio * cos_theta_i - cos_theta_t) * surface_normal

        return refracted_direction

    def reflect(self, direction):
        incident_direction = -direction
        surface_normal = self.normal

        reflected_direction = incident_direction - 2 * np.dot(incident_direction, surface_normal) * surface_normal

        return reflected_direction
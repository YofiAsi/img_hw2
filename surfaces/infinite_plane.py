import numpy as np

class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index

    def intersect(self, origins, directions):
        denom = np.sum(self.normal * directions, axis=1)
        t = -(np.sum(self.normal * origins, axis=1) + self.offset) / denom

        mask = denom != 0
        t = np.where(mask, t, np.inf)
        t = np.maximum(t, 0)

        hit_points = np.where(mask[:, np.newaxis], origins + t[:, np.newaxis] * directions, None)

        return hit_points
    
    def refract(self, directions, intersection_points, refractive_index_ratio=1.5):
        cos_theta_i = -np.sum(directions * self.normal, axis=1)

        mask = cos_theta_i > 0
        surface_normal = np.where(mask[:, np.newaxis], -self.normal, self.normal)
        refractive_index_ratio = np.where(mask, 1 / refractive_index_ratio, refractive_index_ratio)

        cos_theta_t = np.sqrt(1 - refractive_index_ratio**2 * (1 - cos_theta_i**2))
        cos_theta_t = np.where(np.isnan(cos_theta_t), 0, cos_theta_t)

        refracted_directions = refractive_index_ratio[:, np.newaxis] * directions + \
            (refractive_index_ratio[:, np.newaxis] * cos_theta_i[:, np.newaxis] - cos_theta_t[:, np.newaxis]) * surface_normal

        return refracted_directions

    def reflect(self, directions):
        incident_directions = -directions
        surface_normal = self.normal

        reflected_directions = incident_directions - 2 * np.sum(incident_directions * surface_normal, axis=1)[:, np.newaxis] * surface_normal

        return reflected_directions

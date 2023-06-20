import numpy as np

class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = position
        self.radius = radius
        self.material_index = material_index

    def intersect(self, origins, directions):
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

    def refract(self, directions, intersection_points, refractive_index_ratio=1.5):
        surface_normals = (intersection_points - self.position) / self.radius

        cos_theta_i = -np.sum(directions * surface_normals, axis=1)

        mask = cos_theta_i > 0
        surface_normals = np.where(mask[:, np.newaxis], -surface_normals, surface_normals)
        refractive_index_ratio = np.where(mask, 1 / refractive_index_ratio, refractive_index_ratio)

        cos_theta_t = np.sqrt(1 - refractive_index_ratio**2 * (1 - cos_theta_i**2))
        cos_theta_t = np.where(np.isnan(cos_theta_t), 0, cos_theta_t)

        refracted_directions = refractive_index_ratio[:, np.newaxis] * directions + \
            (refractive_index_ratio[:, np.newaxis] * cos_theta_i[:, np.newaxis] - cos_theta_t[:, np.newaxis]) * surface_normals

        return refracted_directions

    def reflect(self, directions, intersection_points):
        incident_directions = -directions
        surface_normals = (intersection_points - self.position) / self.radius

        reflected_directions = incident_directions - 2 * np.sum(incident_directions * surface_normals, axis=1)[:, np.newaxis] * surface_normals

        return reflected_directions

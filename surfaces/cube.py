import numpy as np

class Cube:
    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale
        self.material_index = material_index

    def intersect(self, origins, directions):
        inv_directions = 1.0 / directions
        sign = np.where(inv_directions >= 0, 1, -1)

        tmin = (self.position - origins) * inv_directions
        tmax = (self.position + self.scale - origins) * inv_directions

        tmin, tmax = np.minimum(tmin, tmax), np.maximum(tmin, tmax)
        tmin = np.max(tmin, axis=1)
        tmax = np.min(tmax, axis=1)

        tmin = np.maximum(tmin, 0)

        mask = tmin <= tmax
        hit_points = np.where(mask[:, np.newaxis], origins + tmin[:, np.newaxis] * directions, None)

        return hit_points

    def refract(self, ray_directions, intersection_points, refractive_index_ratio=1.5):
        normals = self._get_surface_normal(intersection_points)
        cos_theta_i = np.sum(-ray_directions * normals, axis=1)
        sin_theta_i = np.sqrt(1 - cos_theta_i**2)
        sin_theta_t = sin_theta_i / refractive_index_ratio
        cos_theta_t = np.sqrt(1 - sin_theta_t**2)

        refracted_directions = refractive_index_ratio[:, np.newaxis] * ray_directions + \
            (refractive_index_ratio[:, np.newaxis] * cos_theta_i[:, np.newaxis] - cos_theta_t[:, np.newaxis]) * normals

        return refracted_directions

    def reflect(self, ray_directions, intersection_points):
        normals = self._get_surface_normal(intersection_points)
        reflected_directions = ray_directions - 2 * np.sum(ray_directions * normals, axis=1)[:, np.newaxis] * normals

        return reflected_directions

    def _get_surface_normal(self, points):
        min_dists = np.min(np.abs(points - self.position - self.scale / 2), axis=1)
        epsilon = 1e-6

        mask = np.abs(min_dists - (self.scale / 2)) < epsilon
        normals = np.zeros_like(points)

        for i in range(3):
            indices = np.logical_and(mask, np.abs(points[:, i] - (self.position[i] + self.scale[i] / 2)) < epsilon)
            normals[indices, i] = np.where(points[indices, i] > self.position[i], 1, -1)

        return normals

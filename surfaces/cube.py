import numpy as np

class Cube:
    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale
        self.material_index = material_index

    def intersect(self, rays):
        origins = rays.origin
        directions = rays.direction

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

    def refract(self, ray_direction, intersection_point, refractive_index_ratio=1.5):
        # Compute the refracted direction based on the Snell's law
        # ray_direction: The incident direction of the ray
        # intersection_point: The point of intersection on the cube's surface
        # refractive_index_ratio: The ratio of refractive indices (n1/n2)

        normal = self._get_surface_normal(intersection_point)
        cos_theta_i = np.dot(-ray_direction, normal)
        sin_theta_i = np.sqrt(1 - cos_theta_i**2)
        sin_theta_t = sin_theta_i / refractive_index_ratio
        cos_theta_t = np.sqrt(1 - sin_theta_t**2)

        refracted_direction = refractive_index_ratio * ray_direction + \
            (refractive_index_ratio * cos_theta_i - cos_theta_t) * normal

        return refracted_direction

    def reflect(self, ray_direction, intersection_point):
        # Compute the reflected direction based on the surface normal
        # ray_direction: The incident direction of the ray
        # intersection_point: The point of intersection on the cube's surface

        normal = self._get_surface_normal(intersection_point)
        reflected_direction = ray_direction - 2 * np.dot(ray_direction, normal) * normal

        return reflected_direction

    def _get_surface_normal(self, point):
        # Compute the surface normal at the given point on the cube's surface
        # point: The point on the cube's surface

        # Check which face of the cube the point lies on
        min_dist = np.min(np.abs(point - self.position - self.scale / 2))
        epsilon = 1e-6  # Small value to handle floating-point errors

        if np.abs(min_dist - (self.scale / 2)) < epsilon:
            # Point lies on one of the faces
            normal = np.zeros_like(point)

            for i in range(3):
                if np.abs(point[i] - (self.position[i] + self.scale[i] / 2)) < epsilon:
                    normal[i] = 1 if point[i] > self.position[i] else -1
                    break

            return normal

        # Point does not lie on any face, so return the cube's default normal
        return np.array([0, 0, 0])
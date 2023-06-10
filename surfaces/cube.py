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
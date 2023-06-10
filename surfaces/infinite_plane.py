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
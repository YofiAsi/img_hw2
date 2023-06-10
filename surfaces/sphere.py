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
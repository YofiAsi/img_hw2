import numpy as np

class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index

    def intersect(self, ray):
        dot_product = np.dot(ray.direction, self.normal)
        if dot_product == 0:
            return -1
        return (-1 * (np.dot(ray.origin, self.normal) - self.offset)) / dot_product

    def reflect(self, ray, hit_point):
        reflected_direction = ray.direction - 2 * np.dot(ray.direction, self.normal) * self.normal
        return reflected_direction

    def refract(self, ray, hit_point, refractive_index):
        # Infinite plane does not refract light, so return the same direction
        return ray.direction
    
    def calc_normal(self, point):
        return self.normal
import numpy as np

class Cube:
    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale
        self.material_index = material_index

    def intersect(self, ray):
        t_min = float('-inf')
        t_max = float('inf')

        for i in range(3):
            t1 = (self.position[i] - self.scale[i] - ray.origin[i]) / ray.direction[i]
            t2 = (self.position[i] + self.scale[i] - ray.origin[i]) / ray.direction[i]
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))

        if t_max >= max(0, t_min):
            t = max(0, t_min)
            return t

        return float('-inf')

    def reflect(self, ray, hit_point):
        normal = self.calculate_normal(hit_point)
        reflected_direction = ray.direction - 2 * np.dot(ray.direction, normal) * normal
        return reflected_direction

    def refract(self, ray, hit_point, refractive_index):
        # Cube does not refract light, so return the same direction
        return ray.direction

    def calc_normal(self, point):
        normal = np.zeros(3)
        for i in range(3):
            if abs(point[i] - self.position[i] - self.scale[i]) < 1e-6:
                normal[i] = 1
            elif abs(point[i] - self.position[i] + self.scale[i]) < 1e-6:
                normal[i] = -1
        return normal
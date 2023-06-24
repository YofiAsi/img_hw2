import numpy as np
import ray_tracer

class Cube:
    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale
        self.material_index = material_index

    def intersect(self, ray):
        con_center = self.scale / 2
        min_pos = [p - con_center for p in self.position]
        max_pos = [p + con_center for p in self.position]

        # scale of x indicate the ray x position. x-min <= 1 and x-max >=1 => x of ray close to the box
        x_min = (min_pos[0] - ray.origin[0]) / ray.direction[0]
        x_max = (max_pos[0] - ray.origin[0]) / ray.direction[0]

        x_min, x_max = Cube.swap(x_min, x_max)

        y_min = (min_pos[1] - ray.origin[1]) / ray.direction[1]
        y_max = (max_pos[1] - ray.origin[1]) / ray.direction[1]

        y_min, y_max = Cube.swap(y_min, y_max)

        z_min = (min_pos[2] - ray.origin[2]) / ray.direction[2]
        z_max = (max_pos[2] - ray.origin[2]) / ray.direction[2]

        z_min, z_max = Cube.swap(z_min, z_max)

        # check if the ray out from the cube
        if x_min > y_max or y_min > x_max:
            return -1

        if y_min > x_min:
            x_min = y_min
        if y_max < x_max:
            x_max = y_max

        # check if the ray out from the cube
        if x_min > z_max or z_min > x_max:
            return -1

        if z_min > x_min:
            x_min = z_min
        
        return x_min

    def swap(a, b):
        if a > b:
            return b, a
        return a, b

    def reflect(self, ray, hit_point):
        normal = self.calc_normal(hit_point)
        reflected_direction = ray.direction - 2 * np.dot(ray.direction, normal) * normal
        return reflected_direction

    def refract(self, ray, hit_point, refractive_index):
        # Cube does not refract light, so return the same direction
        return ray.direction

    def calc_normal(self, point):
        con_center = self.scale / 2
        normal = np.zeros(3)

        # intersection is on the upper x-parallel plane
        if abs((point[0] - self.position[0]) - con_center) < ray_tracer.EPSILON:
            normal[0] = 1
        # intersection is on the lower x-parallel plane
        elif abs((self.position[0] - point[0]) - con_center) < ray_tracer.EPSILON:
            normal[0] = -1
        # intersection is on the upper y-parallel plane
        elif abs((point[1] - self.position[1]) - con_center) < ray_tracer.EPSILON:
            normal[1] = 1
        # intersection is on the lower y-parallel plane
        elif abs((self.position[1] - point[1]) - con_center) < ray_tracer.EPSILON:
            normal[1] = -1
        # intersection is on the upper z-parallel plane
        elif abs((point[2] - self.position[2]) - con_center) < ray_tracer.EPSILON:
            normal[2] = 1
        # intersection is on the lower z-parallel plane
        else:
            normal[2] = -1
        
        return normal
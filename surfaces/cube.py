import numpy as np
from math import sqrt
EPSILON = 1e-9

swap = lambda a, b: (b, a) if a > b else (a, b)

class Cube:
    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale
        self.material_index = material_index

    def intersect(self, ray):
        con_center = self.scale / 2
        
        min_pos = self.position - np.array([con_center, con_center, con_center])
        max_pos = self.position + np.array([con_center, con_center, con_center])

        # scale of x indicate the ray x position. x-min <= 1 and x-max >=1 => x of ray close to the box
        if ray.direction[0] == 0:
            if ray.origin[0] < min_pos[0] or ray.origin[0] > max_pos[0]:
                return -1
            x_min = float('inf') * (min_pos[0] - ray.origin[0])
            x_max = float('inf') * (max_pos[0] - ray.origin[0])
        else:
            x_min = (min_pos[0] - ray.origin[0]) / ray.direction[0]
            x_max = (max_pos[0] - ray.origin[0]) / ray.direction[0]

        x_min, x_max = swap(x_min, x_max)

        if ray.direction[1] == 0:
            if ray.origin[1] < min_pos[1] or ray.origin[1] > max_pos[1]:
                return -1
            y_min = float('inf') * (min_pos[1] - ray.origin[1])
            y_max = float('inf') * (max_pos[1] - ray.origin[1])
        else:
            y_min = (min_pos[1] - ray.origin[1]) / ray.direction[1]
            y_max = (max_pos[1] - ray.origin[1]) / ray.direction[1]

        y_min, y_max = swap(y_min, y_max)

        if ray.direction[2] == 0:
            if ray.origin[2] < min_pos[2] or ray.origin[2] > max_pos[2]:
                return -1
            z_min = float('inf') * (min_pos[2] - ray.origin[2])
            z_max = float('inf') * (max_pos[2] - ray.origin[2])
        else:
            z_min = (min_pos[2] - ray.origin[2]) / ray.direction[2]
            z_max = (max_pos[2] - ray.origin[2]) / ray.direction[2]

        z_min, z_max = swap(z_min, z_max)

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

    def reflect(self, ray, hit_point):
        normal = self.calc_normal(hit_point)
        reflected_direction = ray.direction - 2 * np.dot(ray.direction, normal) * normal
        return reflected_direction

    def refract(self, ray, hit_point):
        normal = self.calc_normal(hit_point)
        incident_direction = ray.direction
        refractive_index = 1.08

        cos_i = -np.dot(normal, incident_direction)
        sin_t_squared = refractive_index**2 * (1 - cos_i**2)

        if sin_t_squared > 1.0:
            # Total internal reflection, reflect the ray
            reflected_direction = incident_direction - 2 * np.dot(incident_direction, normal) * normal
            return hit_point, reflected_direction
        else:
            cos_t = sqrt(1 - sin_t_squared)
            refracted_direction = refractive_index * incident_direction + (refractive_index * cos_i - cos_t) * normal
            return hit_point, refracted_direction

    def calc_normal(self, point):
        con_center = self.scale / 2
        normal = np.zeros(3)

        # intersection is on the upper x-parallel plane
        if abs((point[0] - self.position[0]) - con_center) < EPSILON:
            normal[0] = 1
        # intersection is on the lower x-parallel plane
        elif abs((self.position[0] - point[0]) - con_center) < EPSILON:
            normal[0] = -1
        # intersection is on the upper y-parallel plane
        elif abs((point[1] - self.position[1]) - con_center) < EPSILON:
            normal[1] = 1
        # intersection is on the lower y-parallel plane
        elif abs((self.position[1] - point[1]) - con_center) < EPSILON:
            normal[1] = -1
        # intersection is on the upper z-parallel plane
        elif abs((point[2] - self.position[2]) - con_center) < EPSILON:
            normal[2] = 1
        # intersection is on the lower z-parallel plane
        else:
            normal[2] = -1
        
        return normal
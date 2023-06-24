import numpy as np

EPSILON = 1e-9

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

    def refract(self, ray, hit_point, refractive_index=1.5):
            incident_direction = ray.direction
            normal = np.array([0, 0, 1])  # Assuming the cube's normal is along the z-axis

            # Calculate the dot product between the incident direction and the normal
            dot_product = np.dot(incident_direction, normal)

            if dot_product < 0:
                # Ray is entering the cube
                refractive_ratio = 1 / refractive_index
                new_normal = normal
            else:
                # Ray is exiting the cube
                refractive_ratio = refractive_index
                new_normal = -normal

            # Calculate the refracted direction using Snell's law
            cos_theta_i = -dot_product
            sin_theta_i = np.sqrt(max(0, 1 - cos_theta_i ** 2))
            sin_theta_t = refractive_ratio * sin_theta_i

            if sin_theta_t >= 1:
                # Total internal reflection
                refracted_direction = None
            else:
                cos_theta_t = np.sqrt(max(0, 1 - sin_theta_t ** 2))
                refracted_direction = refractive_ratio * incident_direction + (refractive_ratio * cos_theta_i - cos_theta_t) * new_normal

            return refracted_direction

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
import argparse
import numpy as np
from random import random
from numpy import ndarray
from numpy.linalg import norm
from PIL import Image
from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from pathlib import Path

EPSILON = 1e-9


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array, file_path):
    image = Image.fromarray(np.uint8(image_array))
    
    # Save the image to a file
    image.save(file_path)


class Ray:
    def __init__(self, origin: ndarray, direction: ndarray, relfective_depth=0):
        assert len(origin) == 3 and len(direction) == 3
        self.origin = origin
        self.direction = direction
        self.reflective_depth = relfective_depth


class Scene:
    def __init__(self, scene_settings: SceneSettings, objects: list, camera: Camera):
        self.bg_color = np.array(scene_settings.background_color)
        self.n_shadow_rays = scene_settings.root_number_shadow_rays
        self.max_recursions = scene_settings.max_recursions
        
        self.materials = [None] + [obj for obj in objects if isinstance(obj, Material)]
        self.lights = [obj for obj in objects if isinstance(obj, Light)]
        self.objects = [obj for obj in objects if not isinstance(obj, Light) and not isinstance(obj, Material)]
        
        self.camera_position = camera.position


def calculate_perpendicular_vector(vector):
    if np.allclose(vector, [0., 0., 0.]):
        raise ValueError("Zero vector does not have a unique perpendicular vector.")

    if np.allclose(vector[:2], [0., 0.]):
        # Vector lies along the z-axis, return a perpendicular vector lying in the x-y plane
        return np.array([1., 0., 0.]) if vector[2] != 0 else np.array([0., 1., 0.])

    # Generate two non-parallel vectors by changing the first component
    v1 = np.array([-vector[1], vector[0], 0])
    v2 = np.array([-vector[2], 0, vector[0]])

    # Choose the shorter vector as the perpendicular vector
    if np.linalg.norm(v1) < np.linalg.norm(v2):
        return v1 / np.linalg.norm(v1)
    else:
        return v2 / np.linalg.norm(v2)

def find_all_intersections(ray: Ray, objects: list):
    intersections = []

    for obj in objects:
        t = obj.intersect(ray)

        if EPSILON <= t:
            intersections.append((t, obj))
    
    # sort the intersections by t
    intersections.sort(key=lambda x: x[0])

    return intersections

def find_nearest_intersection(ray: Ray, objects: list):
    min_t = float('inf')
    intersected_object = None

    for obj in objects:
        t = obj.intersect(ray)

        if EPSILON <= t < min_t:
            min_t = t
            intersected_object = obj
    
    return min_t, intersected_object

def normalize(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude <= EPSILON:
        return vector
    return vector / magnitude  
    
def set_camera_orientation(camera: Camera):
    P_0 = camera.position
    towards_vector = normalize(np.array(camera.look_at) - np.array(P_0))
    right_vector = normalize(np.cross(towards_vector, camera.up_vector))
    up_vector = normalize(np.cross(right_vector, towards_vector))

    camera.right_vector = right_vector
    camera.up_vector = up_vector
    camera.towards_vector = towards_vector

def construct_ray_grid(camera: Camera, image_width: int, image_height: int):
    """
    This is the first step. We construct a grid of rays, one for each pixel in the image.
    """

    # Calculate the position of each pixel on the 3D world
    P_center = camera.position + camera.screen_distance * camera.towards_vector
    R = camera.screen_width / image_width

    P = np.zeros((image_height, image_width, 3))

    for i in range(image_height):
        for j in range(image_width):
            P[i, j] = P_center + R * (j - image_width / 2) * camera.right_vector - R * (i - image_height / 2) * camera.up_vector

    # Calculate the direction of each ray:
    directions = np.zeros((image_height, image_width, 3))
    directions[:, :] = P - camera.position
    directions /= np.linalg.norm(directions, axis=-1)[..., None]

    ray_origins = np.full((image_height, image_width, 3), camera.position)

    ray_grid = np.array([[Ray(ray_origins[i, j], directions[i, j]) for j in range(image_width)] for i in range(image_height)])

    return ray_grid

def calc_light_intensity(scene: Scene, light: Light, intersection_point: ndarray, intersected_object):
    N = int(scene.n_shadow_rays)
    light_vector: ndarray = normalize(intersection_point - light.position)

    # Create perpendicular plane x,y to ray
    x = calculate_perpendicular_vector(light_vector)
    y = normalize(np.cross(light_vector, x))

    # Create rectangle
    left_bottom_cell = light.position - (light.radius/2) * x - (light.radius/2) * y

    # Normalize rectangle directions by cell size:
    cell_length = light.radius / N
    x *= cell_length
    y *= cell_length

    # Cast ray from cell to point and see if intersect with our point first
    intersect_counter = 0.
    for i in range(N):
        for j in range(N):
            cell_pos = left_bottom_cell + (i + random()) * x + (j + random()) * y
            ray_vector = normalize(intersection_point - cell_pos)
            cell_light_ray = Ray(cell_pos, ray_vector)
            intersections = find_all_intersections(cell_light_ray, scene.objects)
            # cell_t, cell_obj = find_nearest_intersection(cell_light_ray, scene.objects)

            # checks if cell intersects with our point first
            transparency_val = 1.

            for t, obj in intersections:
                if obj == intersected_object:
                    intersect_counter += 1.
                    break
                if scene.materials[obj.material_index].transparency == 0:
                    break
                transparency_val *= scene.materials[obj.material_index].transparency
                if transparency_val < EPSILON:
                    break

            # if cell_obj == intersected_object:
            #     intersect_counter += 1.
            
    fraction = float(intersect_counter) / float(N * N)
    return (1 - light.shadow_intensity) + (light.shadow_intensity * fraction * transparency_val)

def calc_diffuse_color(light: Light, light_intens, intersection_point: ndarray, normal: ndarray):
    light_vector = normalize(light.position - intersection_point)
    dot_product = np.dot(normal, light_vector)
    if dot_product < 0:
        return np.zeros(3, dtype=float)
    diffuse = light.color * light_intens * dot_product
    return diffuse

def calc_specular_color(light: Light, camera_pos: ndarray, light_intens, intersection_point: ndarray, normal: ndarray, shininess: int):
    L = normalize(intersection_point - light.position)
    R = normalize(L - (2 * np.dot(L, normal) * normal))
    V = normalize(camera_pos - intersection_point)
    dot_product = np.dot(R, V)
    if dot_product < 0:
        return np.zeros(3, dtype=float)
    specular = light.color * light_intens * (dot_product ** shininess) * light.specular_intensity
    return specular

def calc_lighting(intersection_point: ndarray, intersected_object, material: Material, scene: Scene):
    camera_pos = scene.camera_position
    shininess = material.shininess
    normal = normalize(intersected_object.calc_normal(intersection_point))
    diffuse_color = np.zeros(3, dtype=float)
    specular_color = np.zeros(3, dtype=float)

    for light in scene.lights:
        # Calculate the light intensity
        light_intens = calc_light_intensity(scene, light, intersection_point, intersected_object)
        # Compute light effect on diffuse color
        diffuse_color += calc_diffuse_color(light, light_intens, intersection_point, normal)
        # Compute light effect on specular color
        specular_color += calc_specular_color(light, camera_pos, light_intens, intersection_point, normal, shininess)

    return diffuse_color, specular_color

def calc_ray_color(ray: Ray, scene: Scene):
    # recurtion stop condition
    if ray.reflective_depth >= scene.max_recursions:
        return np.copy(scene.bg_color)
    
    # find the nearest intersection
    t, intersected_object = find_nearest_intersection(ray, scene.objects)

    # if there is no intersection, return the background color
    if intersected_object is None:
        return np.copy(scene.bg_color)

    # calculate the intersection point and get material properties
    intersection_point: ndarray = ray.origin + t * ray.direction
    material: Material = scene.materials[intersected_object.material_index]
    specular: ndarray =  material.specular_color
    diffuse: ndarray = material.diffuse_color
    mat_reflection_color: ndarray = material.reflection_color
    transparency: float = material.transparency

    # calculate the reflection ray
    reflection_ray_color = np.zeros(3)
    if (mat_reflection_color != 0).any():
        direction = intersected_object.reflect(ray, intersection_point)
        origin = intersection_point + EPSILON * direction
        reflection_ray = Ray(origin, direction, ray.reflective_depth + 1)
        reflection_ray_color = calc_ray_color(reflection_ray, scene)
    
    # calculate the refraction ray (transparency)
    refraction_ray_color = np.zeros(3)
    if transparency > 0:
        origin, direction = intersected_object.refract(ray, intersection_point)
        origin = intersection_point + EPSILON * direction
        refraction_ray = Ray(origin, direction, 0)
        refraction_ray_color = np.copy(calc_ray_color(refraction_ray, scene))

    # calculate the diffuse and specular colors
    diffuse_color, specular_color = calc_lighting(intersection_point, intersected_object, material , scene)
    diffuse_color = np.copy(diffuse_color)
    specular_color = np.copy(specular_color)

    # combine the colors
    diffuse_color *= diffuse
    specular_color *= specular
    reflection_ray_color *= mat_reflection_color

    return refraction_ray_color * transparency + (diffuse_color + specular_color) * (1 - transparency) + reflection_ray_color

def ray_trace(ray_grid: ndarray, scene: Scene):
    color_matrix = np.zeros((ray_grid.shape[0], ray_grid.shape[1], 3))

    for i in range(ray_grid.shape[0]):
        for j in range(ray_grid.shape[1]):
            color_matrix[i, j] = np.clip(calc_ray_color(ray_grid[i, j], scene) * 255, 0, 255)

    return color_matrix



def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # Get the camera orientation
    set_camera_orientation(camera)

    # Set up the scene
    scene = Scene(scene_settings, objects, camera)

    # Construct the ray grid
    ray_grid = construct_ray_grid(camera, args.width, args.height)

    # Trace the rays
    color_matrix = ray_trace(ray_grid, scene)

    # Save the output image
    save_image(color_matrix, args.output_image)

if __name__ == '__main__':
    main()

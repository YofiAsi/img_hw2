import argparse
import numpy as np
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
from tqdm import tqdm

EPSILON = 1e-6

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
    def __init__(self, origin: ndarray, direction: ndarray, origin_object=None, relfective_depth=0):
        assert len(origin) == 3 and len(direction) == 3
        self.origin = origin
        self.direction = direction
        self.color = np.array([0, 0, 0])
        self.transparency_ray:  Ray = None
        self.reflection_ray:    Ray = None
        self.intersected_object = None
        self.intersecting_point = None
        self.reflective_depth = 0

class Vector:
    def __init__(self, direction):
        """
        create a normalized vector
        """
        magnitude = np.linalg.norm(direction)
        if (direction == 0).all():
            self.direction = direction
        else:
            self.direction = direction / magnitude

    def get_perpendicular_vector(self):
        vector = np.cross(self.direction, np.array([1, 0, 0]))
        if (vector == 0).all():
            vector = np.cross(self.direction, np.array([0, 1, 0]))
        vector /= np.linalg.norm(vector)
        return vector
    
def set_camera_orientation(camera: Camera):
    P_0 = camera.position
    towards_vector = Vector(np.array(camera.look_at) - np.array(P_0))
    right_vector = Vector(np.cross(towards_vector.direction, camera.up_vector))
    up_vector = Vector(np.cross(right_vector.direction, towards_vector.direction))

    camera.right_vector = right_vector.direction
    camera.up_vector = up_vector.direction
    camera.towards_vector = towards_vector.direction

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

def refract(incident_direction, surface_normal, refractive_index_ratio):
    cos_theta_i = -np.dot(incident_direction, surface_normal)
    sin2_theta_i = 1 - cos_theta_i**2

    if sin2_theta_i > 1:
        # Total internal reflection
        return None

    cos_theta_t = np.sqrt(1 - refractive_index_ratio**2 * sin2_theta_i)
    refracted_direction = refractive_index_ratio * incident_direction + (refractive_index_ratio * cos_theta_i - cos_theta_t) * surface_normal

    return refracted_direction


def construct_next_ray_grid(curr_ray_array: ndarray, intersections: ndarray, object_hits: ndarray, material_list: list) -> ndarray:
    new_rays = []  # List to store the new rays

    # Filter valid rays based on intersections and object hits
    valid_indices = np.logical_and(intersections != None, object_hits != None)
    valid_rays = curr_ray_array[valid_indices]

    valid_intersections = intersections[valid_indices]
    valid_object_hits = object_hits[valid_indices]

    # Extract object materials based on material index from valid object hits
    object_materials = np.take(material_list, [obj_hit.material_index for obj_hit in valid_object_hits])

    # Calculate reflection colors and filter reflection mask
    reflection_colors = np.vectorize(lambda x: x.reflection_color)(object_materials)
    reflection_mask = np.any(reflection_colors != [0, 0, 0], axis=1)

    # Calculate reflection directions, origins, and create reflection rays
    reflection_directions = np.array([obj_hit.reflect(ray.direction, intersection) for ray, obj_hit, intersection in zip(valid_rays, valid_object_hits, valid_intersections)])
    reflection_origins = valid_intersections + reflection_directions * EPSILON
    reflection_rays = Ray(reflection_origins, reflection_directions)
    reflection_rays.color = reflection_colors.reshape(-1)
    reflection_rays.origin_object = valid_object_hits
    reflection_rays.reflective_depth = valid_rays.reflective_depth + 1

    # Check transparency mask
    transparency_mask = object_materials[:, 0] != 0

    # Calculate transparency directions, origins, and create transparency rays
    transparency_directions = np.array([refract(ray.direction, obj_hit.normal, obj_mat.refraction_index) for ray, obj_hit, obj_mat in zip(valid_rays, valid_object_hits, object_materials)])
    transparency_origins = valid_intersections + transparency_directions * EPSILON
    transparency_rays = Ray(transparency_origins, transparency_directions)
    transparency_rays.color = valid_rays.color
    transparency_rays.origin_object = valid_object_hits

    # Extend new rays with valid reflection rays
    if np.any(reflection_mask):
        new_rays.extend(reflection_rays[reflection_mask])

    # Extend new rays with valid transparency rays
    if np.any(transparency_mask):
        new_rays.extend(transparency_rays[transparency_mask])

    new_ray_array = np.array(new_rays, dtype=np.object)  # Convert the list of new rays to an array

    return new_ray_array


def find_nearest_intersection(ray: Ray, objects: list):
    min_t = float('inf')
    
    for obj in objects:
        t = obj.intersect(ray)

        if t is None:
            continue

        # distance = np.linalg.norm(intersection - ray.origin)

        if EPSILON <= t < min_t:
            min_t = t
            ray.intersecting_point = ray.origin + t * ray.direction
            ray.intersected_object = obj

def get_ray_color(ray: Ray, objects: list, materials: list, lights: list, scene_settings: SceneSettings):
    if ray.intersected_object is None:
        ray.color = scene_settings.background_color
        return
    
    material = materials[ray.intersected_object.material_index - 1]
    ray.color = material.diffuse_color

def ray_trace(ray_grid: ndarray, objects_list: list, material_list: list, light_list: list, scene_settings: SceneSettings):
    BG_COLOR = scene_settings.background_color
    MAX_SHADOW_RAYS = scene_settings.root_number_shadow_rays
    MAX_DEPTH = scene_settings.max_recursions

    count = 0
    for ray in tqdm(ray_grid.flatten(), total=ray_grid.size, desc='Ray Tracing'):

        if count == 250000//2 - 10:
            print("hello")

        find_nearest_intersection(ray, objects_list)
        # if ray.intersected_object is not None:
        #     print("hello")
        get_ray_color(
            ray=ray,
            objects=objects_list,
            materials=material_list,
            lights=light_list,
            scene_settings=scene_settings
        )
        count += 1

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

    material_list = [obj for obj in objects if isinstance(obj, Material)]
    light_list = [obj for obj in objects if isinstance(obj, Light)]
    object_list = [obj for obj in objects if not isinstance(obj, Light) and not isinstance(obj, Material)]

    # Construct the ray grid and reshape it to a 1D array
    ray_grid = construct_ray_grid(camera, args.width, args.height)

    ray_trace(ray_grid, object_list, material_list, light_list, scene_settings)

    # create a color matrix
    color_matrix = np.array([[ray.color for ray in row] for row in ray_grid]) * 255

    # Save the output image
    save_image(color_matrix, args.output_image)


if __name__ == '__main__':
    main()

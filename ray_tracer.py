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

EPSILON = 1e-6

class RayArray:
    def __init__(self, origins: ndarray, directions: ndarray, reflective_depth = None):
        assert origins.shape == directions.shape
        self.origins = np.copy(origins)
        self.directions = np.copy(directions)
        self.colors = np.zeros_like(origins)
        self.reflective_depth = reflective_depth
        self.object_hits = None
        self.hit_points = None
        self.reflections = None
        self.transparencies = None

    def get_next_ray_array(self, material_list: list, object_list: list):
        valid_indices = np.logical_and(self.hit_points != None, self.object_hits != None)
        valid_origins = self.origins[valid_indices]
        valid_directions = self.directions[valid_indices]

        valid_intersections = self.hit_points[valid_indices]
        valid_object_hits = self.object_hits[valid_indices]

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
        transparency_directions = np.array([obj_hit.refract(ray.direction, obj_hit.normal, obj_mat.refraction_index) for ray, obj_hit, obj_mat in zip(valid_rays, valid_object_hits, object_materials)])
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

    def get_colors(self, objects_list: list, material_list: list, scene_settings: SceneSettings):
        valid_indices = np.logical_and(self.hit_points != None, self.object_hits != None)
        backgournd_indices = np.logical_not(valid_indices)
        self.colors[backgournd_indices] = scene_settings.background_color

        for object in objects_list:
            object_indices = self.object_hits == object
            material_idx = object.material_index
            material: Material = material_list[material_idx]
            self.colors[object_indices] = material.diffuse_color

class Ray:
    def __int__(self, origin: ndarray, direction: ndarray, origin_object=None, relfective_depth=0):
        assert len(origin) == 3 and len(direction) == 3
        self.origin = origin
        self.direction = direction
        self.color = np.array([0, 0, 0])
        self.transparency_ray:  Ray = None
        self.reflection_ray:    Ray = None
        self.origin_object = None
        self.reflective_depth = 0

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

def normalize(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
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

    j_coords = np.arange(image_width) - np.floor(image_width/2)
    i_coords = np.arange(image_height) - np.floor(image_height/2)

    P = np.zeros(shape=(image_height, image_width, 3))
    P = P + P_center + R * j_coords[:, np.newaxis] * camera.right_vector.reshape(1, -1) - R * i_coords.reshape(-1, 1) * camera.up_vector.reshape(1, -1, 1)
    directions = normalize(P - camera.position)

    ray_grid = RayArray(ray_origins.reshape(-1), directions.reshape(-1))

    return ray_grid

def find_nearest_intersection(ray_grid: ndarray, scene_objects):
    intersections = np.empty_like(ray_grid, dtype=np.object)
    object_hits = np.empty_like(ray_grid, dtype=np.object)
    t_min = np.full(ray_grid.shape, float('inf'))

    for obj in scene_objects:
        hit_points = obj.intersect(ray_grid)
        valid_hits = hit_points != None
        t_values = np.linalg.norm(hit_points - ray_grid.origin[..., None, None], axis=-1)

        update_mask = np.logical_and(valid_hits, t_values < t_min)
        t_min[update_mask] = t_values[update_mask]
        intersections[update_mask] = hit_points[update_mask]
        object_hits[update_mask] = obj

    return intersections, object_hits

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
    transparency_directions = np.array([obj_hit.refract(ray.direction, obj_hit.normal, obj_mat.refraction_index) for ray, obj_hit, obj_mat in zip(valid_rays, valid_object_hits, object_materials)])
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



def ray_trace(ray_array: RayArray, objects: list, material_list: list, light_list: list, scene_settings: SceneSettings):
    BG_COLOR = scene_settings.background_color
    MAX_SHADOW_RAYS = scene_settings.root_number_shadow_rays
    MAX_DEPTH = scene_settings.max_recursions

    rays_stack = [ray_array]
    intersections, object_hits = find_nearest_intersection(ray_array, objects)

    ray_array.get_colors(material_list=material_list, objects_list=objects, scene_settings=scene_settings)

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
    first_ray_array = construct_ray_grid(camera, args.width, args.height)

    ray_trace(first_ray_array, camera, object_list, material_list, light_list, scene_settings)
    
    colors = first_ray_array.colors.reshape((args.width, args.height, 3))

    # Save the output image
    save_image(colors, args.output_image)


if __name__ == '__main__':
    main()

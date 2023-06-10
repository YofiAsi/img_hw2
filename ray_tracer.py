import argparse
from PIL import Image
import numpy as np
from numpy import ndarray
from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
import numpy as np
from numpy.linalg import norm

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
    ray_grid = np.empty((image_height, image_width), dtype=np.object)

    # Calculate the position of each pixel on the 3D world
    P_center = camera.position + camera.screen_distance * camera.towards_vector
    R = camera.screen_width / image_width

    j_coords = np.arange(image_width) - np.floor(image_width/2)
    i_coords = np.arange(image_height) - np.floor(image_height/2)

    P = P_center + R * j_coords[:, np.newaxis] * camera.right_vector - R * i_coords[np.newaxis, :] * camera.up_vector
    directions = normalize(P - camera.position)

    ray_origins = np.full((image_height, image_width), camera.position)
    ray_grid[:, :] = np.vectorize(Ray)(ray_origins, directions)

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

    # Construct the ray grid and reshape it to a 1D array
    ray_grid_2d = construct_ray_grid(camera, args.width, args.height)
    ray_grid = ray_grid_2d.reshape(-1)



    # Save the output image
    save_image(ray_grid_2d.color, args.output_image)


if __name__ == '__main__':
    main()

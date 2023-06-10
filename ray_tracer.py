import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


class Ray:
    def __int__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def point_at_parameter(self, t):
        return self.origin + t * self.direction

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


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def calculate_orthogonal_vector(vector):
    # Create a random vector with the same dimension
    random_vector = np.random.rand(len(vector))

    # Subtract the projection of the random vector onto the given vector
    orthogonal_vector = random_vector - np.dot(random_vector, vector) / np.dot(vector, vector) * vector

    return orthogonal_vector

def normalize(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector
    return vector / magnitude
def construct_ray_through_pixel(camera, i, j, image_resolution_width, image_resolution_height):
    P_0 = camera.position
    towards_vector = normalize(np.array(camera.look_at) - np.array(P_0))
    P_center = P_0 + camera.screen_distance * towards_vector
    up_vector = normalize(calculate_orthogonal_vector(towards_vector))

    right_vector = normalize(np.cross(towards_vector, up_vector))
    up_vector = normalize(np.cross(right_vector, towards_vector))
    R = camera.screen_width / image_resolution_width

    P = P_center + (j - np.floor(image_resolution_width/2)) * R * right_vector - \
        (i - np.floor(image_resolution_height/2) * R * up_vector)


    return P



def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer

    # image resolution
    image_height = args.width
    image_width = args.height

    for i in range(image_width):
        for j in range(image_height):
            ray = construct_ray_through_pixel(camera, i, j, image_width, image_height)



    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()

import functools
import math

import numpy as np
import matplotlib.pyplot  as plt

from genevis.render import RaycastRenderer
from genevis.transfer_function import TFColor
from volume.volume import GradientVolume, Volume
from itertools import permutations
from genevis.transfer_function import TransferFunction

def get_voxelInterpolated(volume: Volume, x: float, y: float, z: float):
    """
    Retrieves the value of a voxel for the given coordinates.
    :param volume: Volume from which the voxel will be retrieved.
    :param x: X coordinate of the voxel
    :param y: Y coordinate of the voxel
    :param z: Z coordinate of the voxel
    :return: Voxel value
    """
    if x < 0 or y < 0 or z < 0 or x >= volume.dim_x or y >= volume.dim_y or z >= volume.dim_z:
        return 0

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    z0 = int(np.floor(z))

    x1 = int(np.floor(x) + 1)
    y1 = int(np.floor(y) + 1)
    z1 = int(np.floor(z) + 1)

    alpha = x - x0 / (x1 - x0)
    beta = y - y0 / (y1 - y0)
    gamma = z - z0 / (z1 - z0)

    vo = get_voxel(volume, x0, y0, z0)
    v1 = get_voxel(volume, x1, y0, z0)
    v2 = get_voxel(volume, x0, y1, z0)
    v3 = get_voxel(volume, x1, y1, z0)
    v4 = get_voxel(volume, x0, y0, z1)
    v5 = get_voxel(volume, x1, y0, z1)
    v6 = get_voxel(volume, x0, y1, z1)
    v7 = get_voxel(volume, x1, y1, z1)

    val = (1 - alpha) * (1 - beta) * (1 - gamma) * vo + \
          alpha * (1 - beta) * (1 - gamma) * v1 + \
          (1 - alpha) * beta * (1 - gamma) * v2 + \
          (alpha) * (beta) * (1 - gamma) * v3 + \
          (1 - alpha) * (1 - beta) * gamma * v4 + \
          alpha * (1 - beta) * gamma * v5 + \
          (1 - alpha) * beta * gamma * v6 + \
          (alpha * gamma * beta) * v7

    return val

def get_voxel(volume: Volume, x: float, y: float, z: float):
    """
    Retrieves the value of a voxel for the given coordinates.
    :param volume: Volume from which the voxel will be retrieved.
    :param x: X coordinate of the               voxel
    :param y: Y coordinate of the voxel
    :param z: Z coordinate of the voxel
    :return: Voxel value
    """
    if x < 0 or y < 0 or z < 0 or x >= volume.dim_x or y >= volume.dim_y or z >= volume.dim_z:
        return 0
    x = int(math.floor(x))
    y = int(math.floor(y))
    z = int(math.floor(z))
    return volume.data[x, y, z]


def compute_gradient(volume: Volume, x: float, y: float, z: float):
    gradient = [0, 0, 0]
    gradient[0] = (get_voxelInterpolated(volume, x + 1, y, z) - get_voxelInterpolated(volume, x - 1, y, z)) / 2
    gradient[1] = (get_voxelInterpolated(volume, x, y + 1, z) - get_voxelInterpolated(volume, x, y - 1, z)) / 2
    gradient[2] = (get_voxelInterpolated(volume, x, y, z + 1) - get_voxelInterpolated(volume, x, y, z - 1)) / 2
    return gradient


class RaycastRendererImplementation(RaycastRenderer):

    def clear_image(self):
        """Clears the image data"""
        self.image.fill(0)

    def render_slicer(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]
        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]
        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2
        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                # Compute the new coordinates in a vectorized form
                voxel_cords = np.dot(u_vector, i - image_center) + np.dot(v_vector, j - image_center) + volume_center

                # Get voxel value
                value = get_voxel(volume, voxel_cords[0], voxel_cords[1], voxel_cords[2])

                # Normalize value to be between 0 and 1
                red = value / volume_maximum
                green = red
                blue = red
                alpha = 1.0 if red > 0 else 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    def render_mip(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3].reshape(-1,1)
        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7].reshape(-1,1)
        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11].reshape(-1,1)

        # Center of the image. Image is squared
        image_center = image_size / 2
        # Center of the volume (3-dimensional)
        volume_center = np.asarray([volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]).reshape(-1,1)
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 10 if self.interactive_mode else 1

        diagonal = (np.sqrt(3) * np.max([volume.dim_x,volume.dim_y,volume.dim_z]))/2
        diagonal = int(math.floor(diagonal)+1)

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                max_voxel_value = []
                for k in range(-diagonal, diagonal, 1):
                    # Compute the new coordinates in a vectorized form
                    voxel_cords = np.dot(u_vector, i-image_center) + np.dot(v_vector, j-image_center) \
                                  + np.dot(view_vector, k) + volume_center

                    max_voxel_value.append(get_voxelInterpolated(volume, voxel_cords[0], voxel_cords[1], voxel_cords[2]))

                value = np.amax(max_voxel_value)

                # Normalize value to be between 0 and 1
                red = value / volume_maximum
                green = red
                blue = red
                alpha = 1.0 if red > 0 else 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    def render_compositing(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray,
                           step=1):
        # Clear the image
        self.clear_image()

        u_vector = view_matrix[0:3].reshape(-1, 1)
        v_vector = view_matrix[4:7].reshape(-1, 1)
        view_vector = view_matrix[8:11].reshape(-1, 1)

        image_center = image_size / 2
        volume_center = np.asarray([volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]).reshape(-1, 1)
        volume_maximum = volume.get_maximum()

        step = 10 if self.interactive_mode else 1

        diagonal = np.sqrt(3) * np.max([volume.dim_x, volume.dim_y, volume.dim_z]) / 2
        diagonal = int(math.floor(diagonal)) + 1

        for i in range(0, int(image_size), step):
            for j in range(0, int(image_size), step):
                red, green, blue, alpha = [0, 0, 0, 1]
                initial_color = TFColor(0, 0, 0, 0)
                for k in range(diagonal, -diagonal, -1):
                    # Compute the new coordinates in a vectorized form
                    voxel_cords = np.dot(u_vector, i - image_center) \
                                  + np.dot(v_vector, j - image_center) \
                                  + np.dot(view_vector, k) + volume_center

                    voxel = get_voxelInterpolated(volume, voxel_cords[0], voxel_cords[1], voxel_cords[2])

                    color = self.tfunc.get_color(voxel)

                    current_color = TFColor(color.a * color.r + (1 - color.a) * initial_color.r,
                                            color.a * color.g + (1 - color.a) * initial_color.g,
                                            color.a * color.b + (1 - color.a) * initial_color.b,
                                            color.a)
                    initial_color = current_color

                red = math.floor(current_color.r * 255) if red < 255 else 255
                green = math.floor(current_color.g * 255) if green < 255 else 255
                blue = math.floor(current_color.b * 255) if blue < 255 else 255
                alpha = math.floor(255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    def render_energy_compositing(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray, energy_volumes: dict):

        # Clear the image
        self.clear_image()

        # Define color dictionary to associate a color to each energy
        perm = np.asarray(list(permutations([1, 0, 0.5, 1], 3)))
        ids = list(energy_volumes.keys())
        colorDictionary = {0: [0, 0, 0]}
        for i in range(len(ids)):
            colorDictionary[ids[i]] = perm[i]

        u_vector = view_matrix[0:3].reshape(-1, 1)
        v_vector = view_matrix[4:7].reshape(-1, 1)
        view_vector = view_matrix[8:11].reshape(-1, 1)

        image_center = image_size / 2
        volume_center = np.asarray([volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]).reshape(-1, 1)
        diagonal = np.sqrt(3) * np.max([volume.dim_x, volume.dim_y, volume.dim_z]) / 2
        diagonal = int(math.floor(diagonal)) + 1

        step = 20 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                red, green, blue, alpha = [0, 0, 0, 1]
                initial_r, initial_g, initial_b = 0, 0, 0
                for k in range(diagonal, -diagonal, -1):
                    # Compute the new coordinates in a vectorized form
                    voxel_cords = np.dot(u_vector, i - image_center) \
                                  + np.dot(v_vector, j - image_center) \
                                  + np.dot(view_vector, k) + volume_center

                    color_voxel = np.asarray([0, 0, 0])
                    for key, value in energy_volumes.items():
                        intensity_max = value.get_maximum()
                        energy_intensity = get_voxelInterpolated(value, voxel_cords[0], voxel_cords[1], voxel_cords[2])
                        if energy_intensity/intensity_max > 0.3:
                            # Show only energy with an intensity above a treshold (30%)
                            color_voxel = np.add(np.multiply(color_voxel, 1-energy_intensity/intensity_max), np.multiply(colorDictionary[key], energy_intensity/intensity_max))
                    color_a = np.max(color_voxel)

                    current_r = color_a * color_voxel[0] + (1 - color_a) * initial_r
                    current_g = color_a * color_voxel[1] + (1 - color_a) * initial_g
                    current_b = color_a * color_voxel[2] + (1 - color_a) * initial_b

                    initial_r, initial_g, initial_b = current_r, current_g, current_b

                red = math.floor(current_r * 255) if red < 255 else 255
                green = math.floor(current_g * 255) if green < 255 else 255
                blue = math.floor(current_b * 255) if blue < 255 else 255
                alpha = math.floor(255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    def render_energy_region_compositing(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray, energy_volumes: dict, magnitudeVolume: Volume):

        # Clear the image
        self.clear_image()

        # Define color dictionary to associate a color to each energy
        perm = np.asarray(list(permutations([1, 0, 0.5, 1], 3)))
        ids = list(energy_volumes.keys())
        colorDictionary = {0: [0, 0, 0]}
        for i in range(len(ids)):
            colorDictionary[ids[i]] = perm[i]

        u_vector = view_matrix[0:3].reshape(-1, 1)
        v_vector = view_matrix[4:7].reshape(-1, 1)
        view_vector = view_matrix[8:11].reshape(-1, 1)

        image_center = image_size / 2
        volume_center = np.asarray([volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]).reshape(-1, 1)
        diagonal = np.sqrt(3) * np.max([volume.dim_x, volume.dim_y, volume.dim_z]) / 2
        diagonal = int(math.floor(diagonal)) + 1

        step = 20 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                red, green, blue, alpha = [0, 0, 0, 1]
                initial_r, initial_g, initial_b = 0, 0, 0
                for k in range(diagonal, -diagonal, -1):
                    # Compute the new coordinates in a vectorized form
                    voxel_cords = np.dot(u_vector, i - image_center) \
                                  + np.dot(v_vector, j - image_center) \
                                  + np.dot(view_vector, k) + volume_center

                    g = get_voxelInterpolated(magnitudeVolume, voxel_cords[0], voxel_cords[1], voxel_cords[2])
                    if g != 0:
                        # Set a white shade to define the region edges
                        color = self.tfunc.get_color(g)
                        color_voxel = np.asarray([color.r, color.g, color.b])
                    else:
                        # Inside the region there is no shade
                        color_voxel = np.asarray([0, 0, 0])

                    if get_voxelInterpolated(volume, voxel_cords[0], voxel_cords[1], voxel_cords[2]) > 0:
                        # Compute colors only inside the regions of interest
                        for key, value in energy_volumes.items():
                            intensity_max = value.get_maximum()
                            energy_intensity = get_voxelInterpolated(value, voxel_cords[0], voxel_cords[1], voxel_cords[2])
                            if energy_intensity/intensity_max > 0.3:
                                # Show only energy with an intensity above a treshold (30%)
                                color_voxel = np.add(np.multiply(color_voxel, 1-energy_intensity/intensity_max), np.multiply(colorDictionary[key], energy_intensity/intensity_max))
                    color_a = np.max(color_voxel)

                    current_r = color_a * color_voxel[0] + (1 - color_a) * initial_r
                    current_g = color_a * color_voxel[1] + (1 - color_a) * initial_g
                    current_b = color_a * color_voxel[2] + (1 - color_a) * initial_b

                    initial_r, initial_g, initial_b = current_r, current_g, current_b

                red = math.floor(current_r * 255) if red < 255 else 255
                green = math.floor(current_g * 255) if green < 255 else 255
                blue = math.floor(current_b * 255) if blue < 255 else 255
                alpha = math.floor(255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    def render_mouse_brain(self, view_matrix: np.ndarray, annotation_volume: Volume, energy_volumes: dict,
                           image_size: int, image: np.ndarray):

        self.tfunc.init(0, math.ceil(self.annotation_gradient_volume.get_max_gradient_magnitude()))
        magnitudeVolume = Volume(self.annotation_gradient_volume.magnitudeVolume)

        # Chose the visulization mode
        option = 1

        if option == 0:
            # Compositing rendering of the region specified in volume file
            self.render_compositing(view_matrix, magnitudeVolume, image_size, image)
        elif option == 1:
            # Compositing rendering of multiple energy in the whole brain
            self.render_energy_compositing(view_matrix, self.annotation_gradient_volume.volume, image_size, image, energy_volumes)
        elif option == 2:
            # Compositing rendering of multiple energy in the region specified in volume file
            self.render_energy_region_compositing(view_matrix, self.annotation_gradient_volume.volume, image_size, image, energy_volumes, magnitudeVolume)
        pass

import numpy as np
import math

class Volume:
    """
    Volume data class.

    Attributes:
        data: Numpy array with the voxel data. Its shape will be (dim_x, dim_y, dim_z).
        dim_x: Size of dimension X.
        dim_y: Size of dimension Y.
        dim_z: Size of dimension Z.
    """

    def __init__(self, array, compute_histogram=True):
        """
        Inits the volume data.
        :param array: Numpy array with shape (dim_x, dim_y, dim_z).
        """

        self.data = array
        self.histogram = np.array([])
        self.dim_x = array.shape[0]
        self.dim_y = array.shape[1]
        self.dim_z = array.shape[2]

        if compute_histogram:
            self.compute_histogram()

    def get_voxel(self, x, y, z):
        """Retrieves the voxel for the """
        return self.data[x, y, z]

    def get_minimum(self):
        return self.data.min()

    def get_maximum(self):
        return self.data.max()

    def compute_histogram(self):
        self.histogram = np.histogram(self.data, bins=np.arange(self.get_maximum() + 1))[0]


class VoxelGradient:
    def __init__(self, gx=0, gy=0, gz=0):
        self.x = gx
        self.y = gy
        self.z = gz
        self.magnitude = math.sqrt(gx * gx + gy * gy + gz * gz)

ZERO_GRADIENT = VoxelGradient()


class GradientVolume:
    def __init__(self, volume):
        self.volume = volume        #volume with id selected and -1 everywhere else
        self.data = []              #array of object VoxelGradient
        self.magnitudeVolume = []   #array of magnitude value
        self.compute()
        self.max_magnitude = -1.0

    def get_gradient(self, x, y, z):
        return self.data[x + self.volume.dim_x * (y + self.volume.dim_y * z)]

    def set_gradient(self, x, y, z, value):
        self.data[x + self.volume.dim_x * (y + self.volume.dim_y * z)] = value

    def get_voxel(self, i):
        return self.data[i]

    def set_volume_voxel(selfs, volume, x, y, z, value):
        if not(x < 0 or y < 0 or z < 0 or x >= volume.shape[0] or y >= volume.shape[1] or z >= volume.shape[2]):
            x = int(math.floor(x))
            y = int(math.floor(y))
            z = int(math.floor(z))
            volume[x, y, z]=value

    def get_volume_voxel(self, volume, x, y, z):

        if x < 0 or y < 0 or z < 0 or x >= volume.shape[0] or y >= volume.shape[1] or z >= volume.shape[2]:
            return 0
        x = int(math.floor(x))
        y = int(math.floor(y))
        z = int(math.floor(z))
        return volume[x, y, z]

    def compute(self):
        """
        Computes the gradient for the current volume
        """
        print("Volume size: ", self.volume.data.shape)

        #define the region IDS we want to see
        #Render the n_greatest region in the dataset
        n_greatest = 5
        unique, frequencies = np.unique(self.volume.data, return_counts=True)
        indexes_feq = np.flip(np.argsort(frequencies))
        regions = np.array([], dtype=int)
        for l in range(n_greatest):
            if unique[indexes_feq[l]]!=0:
                regions = np.append(regions, unique[indexes_feq[l]]).astype(int)

        # Render only region with id 15750 and 16001
        regions = [15750, 16001]

        print("regions: ", regions)

        self.data = [ZERO_GRADIENT] * (self.volume.dim_x * self.volume.dim_y * self.volume.dim_z)
        self.magnitudeVolume = np.zeros(self.volume.data.shape)
        volume = np.copy(self.volume.data)
        booleanMask = np.zeros(self.volume.data.shape)
        booleanMask = np.isin(self.volume.data, regions)

        for x in range(self.volume.dim_x):
            for y in range(self.volume.dim_y):
                for z in range(self.volume.dim_z):
                    prev_gx, prev_gy, prev_gz = 0, 0, 0
                    new_gx, new_gy, new_gz = 0, 0, 0
                    n = 0
                    for k, region in enumerate(regions):

                        x_plus = 1000 if self.get_volume_voxel(volume, x + 1, y, z)==region else 0
                        x_minus = 1000 if self.get_volume_voxel(volume, x - 1, y, z) == region else 0
                        y_plus = 1000 if self.get_volume_voxel(volume, x, y + 1, z) == region else 0
                        y_minus = 1000 if self.get_volume_voxel(volume, x, y - 1, z) == region else 0
                        z_plus = 1000 if self.get_volume_voxel(volume, x, y, z + 1) == region else 0
                        z_minus = 1000 if self.get_volume_voxel(volume, x, y, z - 1) == region else 0

                        gx = (x_plus - x_minus) / 2
                        gy = (y_plus - y_minus) / 2
                        gz = (z_plus - z_minus) / 2
                        if (gx!=0 or gy!=0 or gz!=0):
                            new_gx = (prev_gx * n + gx) / (n + 1)
                            new_gy = (prev_gy * n + gy) / (n + 1)
                            new_gz = (prev_gz * n + gz) / (n + 1)
                            prev_gx, prev_gy, prev_gz = new_gx, new_gy, new_gz
                            n = n+1
                    self.set_gradient(x, y, z, VoxelGradient(prev_gx, prev_gy, prev_gz))
                    self.magnitudeVolume[x, y, z] = self.get_gradient(x, y, z).magnitude

        self.volume = Volume(booleanMask.astype(int))

    def get_max_gradient_magnitude(self):
        if self.max_magnitude < 0:
            gradient = max(self.data, key=lambda x: x.magnitude)
            self.max_magnitude = gradient.magnitude

        return self.max_magnitude
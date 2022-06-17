import os.path as osp
import numpy as np
import h5py
import torch
import torch.utils.data as data


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


def farthest_point_sample(points, num_centroids):
    # numpy
    pc = points.numpy() if torch.is_tensor(points) else points
    npoints, dim = pc.shape
    xyz = pc[:, :3]  # not considering normals if presents
    centroids = np.zeros((num_centroids,))
    distance = np.ones((npoints,)) * 1e10
    farthest = np.random.randint(0, npoints)
    for i in range(num_centroids):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

    centroids = centroids.astype(np.int32)
    return centroids


def random_sample(points, num_points=2048):
    sampled_idxs = np.random.choice(points.shape[0], num_points, replace=num_points > points.shape[0])
    return points[sampled_idxs]


def pc_normalize(pc):
    # unite cube normalization
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class RandomSample(object):
    def __init__(self, num_points=1024):
        self.num_points = num_points

    def __call__(self, points):
        sampled_idxs = np.random.choice(points.shape[0], self.num_points, replace=self.num_points > points.shape[0])
        return points[sampled_idxs]


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class Center(object):
    def __call__(self, points):
        points[:, :3] -= np.mean(points[:, :3], axis=0)
        return points


class Scale(object):
    def __call__(self, points):
        # diag = np.max(np.sqrt(np.sum(points[:, :3] ** 2, axis=1)))  # fxia22 GitHub repo, center_normalize from @antoalli
        diag = np.max(np.absolute(points[:, :3]))  # normalize_scale from @antoalli
        points[:, :3] /= (diag + np.finfo(float).eps)
        return points


class AugmScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class AugmRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class AugmRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class AugmJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
                .normal_(mean=0.0, std=self.std)
                .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class AugmTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation
        return points


class AugmRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert 0 <= max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        pc = points.numpy()

        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return torch.from_numpy(pc).float()


class H5_Dataset(data.Dataset):
    """ Simple H5 dataset """
    def __init__(self, h5_file, num_points, transforms=None):
        super().__init__()
        if not osp.exists(h5_file):
            raise FileNotFoundError(h5_file)
        self.h5_file = h5_file
        self.num_points = num_points
        self.transforms = transforms
        # load from h5 file
        print(f"Reading data from hdf5 file: {self.h5_file}", end='')
        f = h5py.File(self.h5_file, 'r')
        self.datas = np.asarray(f['data'][:], dtype=np.float32)
        self.labels = np.asarray(f['label'][:], dtype=np.int64)
        f.close()
        print(f" datas: {self.datas.shape}, labels: {self.labels.shape}")

    def __getitem__(self, idx):
        point_set = self.datas[idx]
        lbl = self.labels[idx]

        # sampling
        point_set = random_sample(point_set, num_points=self.num_points)

        # unit cube normalization
        point_set = pc_normalize(point_set)

        # data augm
        if self.transforms:
            point_set = self.transforms(point_set)

        return point_set, lbl

    def __len__(self):
        return len(self.labels)

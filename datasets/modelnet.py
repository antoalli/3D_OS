import os
import shlex
import subprocess
import tqdm
from utils.data_utils import *
import lmdb
import msgpack_numpy
import h5py

modelnet40_label_dict = {
    'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6,
    'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13,
    'dresser': 14, 'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19,
    'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25,
    'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32,
    'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}

modelnet10_label_dict = {
    'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'night_stand': 6,
    'sofa': 7, 'table': 8, 'toilet': 9}

############################################
# Closed Set for Modelnet to SONN experiments

SR1 = {
    "chair": 0,
    "bookshelf": 1,
    "door": 2,
    "sink": 3,
    "sofa": 4
}

SR2 = {
    "bed": 0,
    "toilet": 1,
    "desk": 2,
    "monitor": 3,
    "table": 2
}


# these are always OOD samples in cross-domain experiments!
modelnet_set3 = {
    'bathtub': 404,  # 1,  # simil sink???
    'bottle': 404,  # 5,
    'bowl': 404,  # 6,
    'cup': 404,  # 10,
    'curtain': 404,  # 11,
    'plant': 404,  # 26,  # simil bin???
    'flower_pot': 404,  # 15,  # simil bin???
    'vase': 404,  # 37,  # simil bin???
    'guitar': 404,  # 17,
    'keyboard': 404,  # 18,
    'lamp': 404,  # 19,
    'laptop': 404,  # 20,
    'night_stand': 404,  # 23,  # simil table - hard out-of-distrib.?
    'person': 404,  # 24,
    'piano': 404,  # 25,  # simil table - hard out-of-distrib.?
    'radio': 404,  # 27,
    'stairs': 404,  # 31,
    'tent': 404,  # 34,
    'tv_stand': 404,  # 36,  # simil table - hard out-of-distrib.?
}


################################################


class ModelNet(data.Dataset):
    """
    ModelNet40 normal resampled. 10k sampled points for each shape
    """

    def __init__(self,
                 num_points, data_root=None, dataset='modelnet40', transforms=None, train=True, download=True):
        super().__init__()
        assert dataset in ['modelnet40', 'modelnet10']
        self.dataset = dataset
        self.num_points = min(int(1e4), num_points)
        self.transforms = transforms
        self.data_root = data_root
        self._cache = os.path.join(self.data_root, "{}_normal_resampled_cache".format(self.dataset))
        self.num_classes = 10 if dataset == 'modelnet10' else 40

        if not osp.exists(self._cache):
            self.folder = "modelnet40_normal_resampled"
            self.data_dir = os.path.join(self.data_root, self.folder)
            self.url = (
                "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"
            )

            if download and not os.path.exists(self.data_dir):
                zipfile = os.path.join(self.data_root, os.path.basename(self.url))
                subprocess.check_call(
                    shlex.split("curl {} -o {}".format(self.url, zipfile))
                )

                subprocess.check_call(
                    shlex.split("unzip {} -d {}".format(zipfile, self.data_root))
                )

                subprocess.check_call(shlex.split("rm {}".format(zipfile)))

            self.train = train
            self.catfile = os.path.join(self.data_dir, "{}_shape_names.txt".format(self.dataset))
            self.cat = [line.rstrip() for line in open(self.catfile)]
            self.classes = dict(zip(self.cat, range(len(self.cat))))

            os.makedirs(self._cache)

            print("Converted to LMDB for faster dataloading while training")
            for split in ["train", "test"]:
                if split == "train":
                    shape_ids = [
                        line.rstrip()
                        for line in open(
                            os.path.join(self.data_dir, "{}_train.txt".format(self.dataset))
                        )
                    ]
                else:
                    shape_ids = [
                        line.rstrip()
                        for line in open(
                            os.path.join(self.data_dir, "{}_test.txt".format(self.dataset))
                        )
                    ]

                shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids]
                # list of (shape_name, shape_txt_file_path) tuple
                self.datapath = [
                    (
                        shape_names[i],
                        os.path.join(self.data_dir, shape_names[i], shape_ids[i])
                        + ".txt",
                    )
                    for i in range(len(shape_ids))
                ]

                with lmdb.open(
                        osp.join(self._cache, split), map_size=1 << 36
                ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                    for i in tqdm.trange(len(self.datapath)):
                        fn = self.datapath[i]
                        point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)
                        cls = self.classes[self.datapath[i][0]]
                        cls = int(cls)

                        txn.put(
                            str(i).encode(),
                            msgpack_numpy.packb(
                                dict(pc=point_set, lbl=cls), use_bin_type=True
                            ),
                        )

        self._lmdb_file = osp.join(self._cache, "train" if train else "test")
        with lmdb.open(self._lmdb_file, map_size=1 << 36, lock=False) as lmdb_env:  # lock=False for DDP
            self._len = lmdb_env.stat()["entries"]

        self._lmdb_env = None

    def __getitem__(self, idx):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._lmdb_file, map_size=1 << 36, readonly=True, lock=False
            )

        with self._lmdb_env.begin(buffers=True) as txn:
            ele = msgpack_numpy.unpackb(txn.get(str(idx).encode()), raw=False)

        point_set = ele["pc"][:, 0:3]  # skip normals
        lbl = ele["lbl"]

        if self.dataset == 'modelnet10':
            assert ele["lbl"] < 10

        # sampling
        point_set = random_sample(point_set, num_points=self.num_points)

        # unit cube normalization
        point_set = pc_normalize(point_set)

        # data augm
        if self.transforms:
            point_set = self.transforms(point_set)

        return point_set, lbl

    def __len__(self):
        return self._len


class ModelNet40_OOD(data.Dataset):
    """
    ModelNet40 normal resampled. 10k sampled points for each shape
    Not using LMDB cache!
    """

    def __init__(self, num_points, data_root=None, transforms=None, train=True, class_choice="SR1"):
        super().__init__()
        self.whoami = "ModelNet40_OOD"
        self.split = "train" if train else "test"
        self.num_points = min(int(1e4), num_points)
        self.transforms = transforms
        assert isinstance(class_choice, str) and class_choice.startswith('SR'), \
            f"{self.whoami} - class_choice must be SRX name"
        self.class_choice = eval(class_choice)
        assert isinstance(self.class_choice, dict)
        self.num_classes = len(set(self.class_choice.values()))
        # reading data
        self.data_dir = os.path.join(data_root, "modelnet40_normal_resampled")
        if not osp.exists(self.data_dir):
            raise FileNotFoundError(f"{self.whoami} - {self.data_dir} does not exist")
        # cache
        cache_dir = osp.join(self.data_dir, "ood_sets_cache")  # directory containing cache files
        cache_fn = osp.join(cache_dir, f'{class_choice}_{self.split}.h5')  # path to cache file
        if os.path.exists(cache_fn):
            # read from cache file
            print(f"{self.whoami} - Reading data from h5py file: {cache_fn}")
            f = h5py.File(cache_fn, 'r')
            self.datas = np.asarray(f['data'][:])
            self.labels = np.asarray(f['label'][:], dtype=np.int64)
            f.close()
        else:
            # reading from txt files and building cache for next training/evaluation
            split_file = os.path.join(self.data_dir, f"modelnet40_{self.split}.txt")

            # all paths
            shape_ids = [
                line.rstrip()
                for line in open(
                    os.path.join(self.data_dir, split_file)
                )
            ]

            # all names
            shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids]

            # class choice
            chosen_idxs = [index for index, name in enumerate(shape_names) if name in self.class_choice.keys()]
            self.shape_ids = [shape_ids[_] for _ in chosen_idxs]
            self.shape_names = [shape_names[_] for _ in chosen_idxs]
            del shape_ids, shape_names

            # read chosen data samples from disk
            self.datapath = [
                (
                    self.shape_names[i],
                    os.path.join(self.data_dir, self.shape_names[i], self.shape_ids[i])
                    + ".txt",
                )
                for i in range(len(self.shape_ids))
            ]
            self.datas = []
            self.labels = []
            for i in tqdm.trange(len(self.datapath), desc=f"{self.whoami} loading data from txts", dynamic_ncols=True):
                fn = self.datapath[i]
                point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)
                point_set = point_set[:, 0:3]  # remove normals
                category_name = self.shape_names[i]  # 'airplane'
                cls = self.class_choice[category_name]
                self.datas.append(point_set)  # [1, 10000, 3]
                self.labels.append(cls)

            self.datas = np.stack(self.datas, axis=0)  # [num_samples, 10000, 3]
            self.labels = np.asarray(self.labels, dtype=np.int64)  # [num_samples, ]

            # make cache
            if not osp.exists(cache_dir):
                os.makedirs(cache_dir)
            print(f"Saving h5py datataset to: {cache_fn}")

            with h5py.File(cache_fn, "w") as f:
                f.create_dataset(name='data', data=self.datas, dtype=np.float32, chunks=True)
                f.create_dataset(name='label', data=self.labels, dtype=np.int64, chunks=True)

            print(f"{self.whoami} - Cache built for split: {self.split}, set: {self.class_choice} - "
                  f"datas: {self.datas.shape} labels: {self.labels.shape} ")

        print(f"{self.whoami} - "
              f"split: {self.split}, "
              f"categories: {self.class_choice}")

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

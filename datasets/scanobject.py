import torch.utils.data as data
import os.path as osp
from utils.data_utils import *
import h5py

# ALL SONN LABELS
SONN_label_dict = {
    "bag": 0, "bin": 1, "box": 2,
    "cabinet": 3, "chair": 4, "desk": 5,
    "display": 6, "door": 7, "shelf": 8,
    "table": 9, "bed": 10, "pillow": 11,
    "sink": 12, "sofa": 13, "toilet": 14
}

# all SONN categories but merging desk and table to same class
sonn_all = {
    0: 0,  # "bag"
    1: 1,  # "bin"
    2: 2,  # "box"
    3: 3,  # "cabinet"
    4: 4,  # "chair"
    5: 5,  # "desk" (merged with table)
    6: 6,  # "display"
    7: 7,  # "door"
    8: 8,  # "shelf"
    9: 5,  # "table" (merged with desk)
    10: 9,  # "bed"
    11: 10,  # "pillow"
    12: 11,  # "sink"
    13: 12,  # "sofa"
    14: 13  # "toilet"
}

# modelnet_set1 ==> SONN
sonn_2_mdSet1 = {
    4: 0,  # chair
    8: 1,  # shelf
    7: 2,  # door
    12: 3,  # sink
    13: 4  # sofa
}

# modelnet_set2 ==> SONN
sonn_2_mdSet2 = {
    10: 0,  # bed
    14: 1,  # toilet
    5: 2,  # desk
    6: 3,  # display
    9: 2  # table
}

# common ood set
# these are categories with poor mapping between md and sonn
sonn_ood_common = {
    0: 404,  # bag
    1: 404,  # bin
    2: 404,  # box
    3: 404,  # cabinet
    11: 404  # pillow
}


################################
# for real -> real experiments #
################################
SR12 = {
    4: 0,  # chair
    8: 1,  # shelf
    7: 2,  # door
    12: 3,  # sink
    13: 4,  # sofa
    ######
    10: 5,  # bed
    14: 6,  # toilet
    5: 7,  # desk
    9: 7,  # table
    6: 8,  # display
}


SR13 = {
    4: 0,  # chair
    8: 1,  # shelf
    7: 2,  # door
    12: 3,  # sink
    13: 4,  # sofa
    ######
    0: 5,  # bag
    1: 6,  # bin
    2: 7,  # box
    3: 8,  # cabinet
    11: 9  # pillow
}


SR23 = {
    10: 0,  # bed
    14: 1,  # toilet
    5: 2,  # desk
    9: 2,  # table
    6: 3,  # display
    ######
    0: 4,  # bag
    1: 5,  # bin
    2: 6,  # box
    3: 7,  # cabinet
    11: 8,  # pillow
}

################################


def load_h5_data_label(h5_path):
    f = h5py.File(h5_path, 'r')
    curr_data = f['data'][:]
    curr_label = f['label'][:]
    f.close()
    return np.asarray(curr_data), np.asarray(curr_label)


def load_h5_data_label_list(h5_paths):
    curr_data = []
    curr_label = []
    for curr_h5 in h5_paths:
        f = h5py.File(curr_h5, 'r')
        curr_data.extend(f['data'][:])
        curr_label.extend(f['label'][:])
        f.close()
    return np.asarray(curr_data), np.asarray(curr_label)


class ScanObject(data.Dataset):
    def __init__(
            self,
            data_root="/home/antonioa/data",
            sonn_split="main_split",
            h5_file="objectdataset.h5",
            split="train",
            class_choice=None,
            num_points=1024,
            transforms=None):

        self.whoami = "ScanObject"
        assert split in ['train', 'test', 'all']
        self.split = split
        self.data_dir = osp.join(data_root, "ScanObjectNN/h5_files")
        assert osp.exists(self.data_dir), f"{self.whoami} - {self.data_dir} does not exist"
        self.num_points = num_points
        self.transforms = transforms
        self.sonn_split = sonn_split
        self.h5_file = h5_file
        self.class_choice = class_choice

        if self.split == "train":
            h5_file_path = [osp.join(self.data_dir, sonn_split, f"training_{h5_file}")]
        elif self.split == "test":
            h5_file_path = [osp.join(self.data_dir, sonn_split, f"test_{h5_file}")]
        elif self.split == "all":
            h5_file_path = [osp.join(self.data_dir, sonn_split, f"training_{h5_file}"),
                            osp.join(self.data_dir, sonn_split, f"test_{h5_file}")]
        else:
            raise ValueError(f"Wrong SONN split: {self.split}")

        # LOAD ALL DATA IN MEMORY
        if isinstance(h5_file_path, list):
            self.datas, self.labels = load_h5_data_label_list(h5_file_path)
        else:
            self.datas, self.labels = load_h5_data_label(h5_file_path)

        # CLASS CHOICE
        if self.class_choice is not None:
            if isinstance(self.class_choice, str):
                self.class_choice = eval(self.class_choice)
            if not isinstance(self.class_choice, dict):
                raise ValueError(f"{self.whoami} - cannot load conversion dict with class_choice: {class_choice}")

            chosen_idxs = [index for index, value in enumerate(self.labels) if value in self.class_choice.keys()]
            assert len(chosen_idxs) > 0
            self.datas = self.datas[chosen_idxs]
            self.labels = [self.class_choice[self.labels[idx]] for idx in chosen_idxs]
            self.num_classes = len(set(self.class_choice.values()))
        else:
            self.num_classes = len(SONN_label_dict.keys())

        print(f"ScanObject - "
              f"num_points: {self.num_points}, "
              f"sonn_split: {self.sonn_split}, "
              f"h5_suffix: {self.h5_file}, "
              f"split: {self.split}, "
              f"class_choice: {self.class_choice}, "
              f"num samples: {len(self.datas)}")

    def __getitem__(self, index):
        point_set = np.asarray(self.datas[index], dtype=np.float32)
        assert len(point_set) == 2048, "SONN: expected 2048-points input shape"
        label = self.labels[index]

        # sampling
        point_set = random_sample(points=point_set, num_points=self.num_points)

        # unit cube normalization
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # data augm
        if self.transforms:
            point_set = self.transforms(point_set)

        return point_set, label

    def __len__(self):
        return len(self.datas)

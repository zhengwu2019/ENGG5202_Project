from PIL import Image
import os
import os.path
import numpy as np
import pickle
import random
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

class CIFAR100(VisionDataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, sub_set=0, val=False):

        super(CIFAR100, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.sub_set = sub_set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')

        if not self.train and self.sub_set == -1:
            downloaded_list = self.test_list
        else:
            downloaded_list = self.train_list

        # if self.train and self.sub_set != -1:
        #     downloaded_list = self.train_list
        # elif not self.train or self.sub_set==-1:
        #     downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # divide the training set into 5 subsets at the first time; in the training phase, loading 4 of 5 subsets as training set and leave only one as validation set.
        # division code used only once and save the seq numbers to pkl file.
        if sub_set != -1:
            sub_indices_path = "./data/cifar-100-python/subset_indices.pkl"
            num_subsets = 10
            if not os.path.exists(sub_indices_path):
                num_classes = 100
                training_size = 50000
                self.subsets = [[] for i in  range(num_subsets)]  # grouped by 5 sub lists, and each sub list contains 10000 indices.
                if self.data.shape[0] == training_size:
                    subset = []
                    for i in range(num_classes):
                        # For 100 classes, we randomly assign 100 samples from 500 ones to a subset for each class. Then we can get 5 subsets.
                        maski = np.array(self.targets) == i
                        class_indices = np.arange(training_size)[maski]
                        end_indices_set = set(class_indices)
                        for j in range(num_subsets):
                            start_indices_set = end_indices_set
                            sub_class_indices = np.array(np.random.choice(np.array(list(start_indices_set)), size=int(training_size/num_classes/num_subsets), replace=False))
                            self.subsets[j].extend(sub_class_indices)
                            end_indices_set = start_indices_set - set(sub_class_indices)
                for k in range(num_subsets):
                    random.shuffle(self.subsets[k])
                with open(sub_indices_path, 'wb') as f:
                    pickle.dump(self.subsets, f)
                    print("subset indices dumping finished.")

            with open(sub_indices_path, 'rb') as f:
                self.subsets = pickle.load(f)
            if val:  # get the validation subset (only 1 subset, 10,000)
                self.need_indices = self.subsets[sub_set]
            else:    # get the training subsets (4 subsets, 40,000)
                self.need_indices = []
                for n in range(num_subsets):
                    if n == sub_set:
                        continue
                    self.need_indices.extend(self.subsets[n])
            self.data = self.data[self.need_indices]
            self.targets = np.array(self.targets)[self.need_indices].tolist()


            # g = []
            # num = 0
            # for i in range(num_subsets):
            #     q = self.subsets[i]
            #     num += np.array([e in q for e in self.subsets[0]]).sum()
            #     num += np.array([e in q for e in self.subsets[1]]).sum()
            #     num += np.array([e in q for e in self.subsets[2]]).sum()
            #     num += np.array([e in q for e in self.subsets[3]]).sum()
            #     num += np.array([e in q for e in self.subsets[4]]).sum()
            #     print(num)

            # import ipdb; ipdb.set_trace()

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' + ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}  # seems useless. {key(class_name): value(0~99)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

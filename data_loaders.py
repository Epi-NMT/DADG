import os
import pandas as pd
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import Dataset
import torch.utils.data as Data
from PIL import Image


class _Dataset(Dataset):
    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_arr = np.asarray(self.file.iloc[:, 1])
        self.transform = transforms.Compose([
                                             transforms.Resize((224, 224)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(([0.485, 0.456, 0.406]), ([0.229, 0.224, 0.225]))
                                            ])

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.label_arr[idx]
        sample = image


        if self.transform:
            sample = self.transform(sample)
        return sample, label

class OH_Loader():
    def __init__(self, flags):

        self.img_path = './data/Office_Home/IMG'
        self.source = flags.source
        self.target = flags.target
        # random pick one domain in source domain as meta-test domain
        self.val = random.sample(self.source, 1)
        self.meta_train = list(set(self.source) - set(self.val))
        self.configuration(flags)
    def configuration(self, flags):
        self.train_batchsz = flags.train_batchsz
        self.meta_test_batchsz = flags.meta_test_batchsz
        self.test_batchsz = flags.test_batchsz

    def loader(self, domain):
        file = './data/Office_Home/txt/{}.txt'.format(domain)
        data = _Dataset(file, self.img_path)
        out_loader = Data.DataLoader(data, batch_size=self.train_batchsz, shuffle=True, num_workers=0)
        return out_loader

    def train_loaders(self):

        rand_number = self.source.index(self.val[0])
        train_loader1 = self.loader(self.meta_train[0])
        train_loader2 = self.loader(self.meta_train[1])
        meta_test_loader = self.loader(self.val[0])

        return train_loader1, train_loader2, meta_test_loader, rand_number

    def test_loader(self):

        test_file = './data/Office_Home/txt/{}.txt'.format(self.target)
        test = _Dataset(test_file, self.img_path)
        test_loader = Data.DataLoader(test, batch_size=self.test_batchsz, shuffle=True, num_workers=0)

        return test_loader

class PACS_Loader():
    def __init__(self, flags):

        self.img_path = './data/kfold'
        self.source = flags.source
        self.target = flags.target
        # random pick one domain in source domain as meta-test domain
        self.val = random.sample(self.source, 1)
        self.meta_train = list(set(self.source) - set(self.val))
        self.configuration(flags)

    def configuration(self, flags):
        self.train_batchsz = flags.train_batchsz
        self.meta_test_batchsz = flags.meta_test_batchsz
        self.test_batchsz = flags.test_batchsz

    def loader(self, domain):
        file = './data/txt/{}.txt'.format(domain)
        data = _Dataset(file, self.img_path)
        out_loader = Data.DataLoader(data, batch_size=self.train_batchsz, shuffle=True, num_workers=0)
        return out_loader

    def train_loaders(self):

        rand_number = self.source.index(self.val[0])
        train_loader1 = self.loader(self.meta_train[0])
        train_loader2 = self.loader(self.meta_train[1])
        meta_test_loader = self.loader(self.val[0])

        return train_loader1, train_loader2, meta_test_loader, rand_number

    def test_loader(self):

        test_file = './data/txt/{}.txt'.format(self.target)
        test = _Dataset(test_file, self.img_path)
        test_loader = Data.DataLoader(test, batch_size=self.test_batchsz, shuffle=True, num_workers=0)
        return test_loader

class VLCS_Loader():

    def __init__(self, flags):

        self.img_path = './data/DG/VLCS/IMG'
        self.source = flags.source
        self.target = flags.target
        # random pick one domain in source domain as meta-test domain
        self.val = random.sample(self.source, 1)
        self.meta_train = list(set(self.source) - set(self.val))
        self.configuration(flags)

    def configuration(self, flags):
        self.train_batchsz = flags.train_batchsz
        self.meta_test_batchsz = flags.meta_test_batchsz
        self.test_batchsz = flags.test_batchsz

    def loader(self, domain):
        file = './data/VLCS/txt_file/{}_train.txt'.format(domain)
        data = _Dataset(file, self.img_path)
        out_loader = Data.DataLoader(data, batch_size=self.train_batchsz, shuffle=True, num_workers=0)
        return out_loader

    def train_loaders(self):

        rand_number = self.source.index(self.val[0])
        train_loader1 = self.loader(self.meta_train[0])
        train_loader2 = self.loader(self.meta_train[1])
        meta_test_loader = self.loader(self.val[0])
        return train_loader1, train_loader2, meta_test_loader, rand_number

    def test_loader(self):

        test_file = './data/VLCS/txt_file/{}_train.txt'.format(self.target)
        test = _Dataset(test_file, self.img_path)
        test_loader = Data.DataLoader(test, batch_size=self.test_batchsz, shuffle=True, num_workers=0)
        return test_loader

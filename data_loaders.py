import os
from skimage import io
import pandas as pd
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import Dataset
import torch.utils.data as Data
from PIL import Image


class _Dataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.file = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.label_arr = np.asarray(self.file.iloc[:, 1])
        self.transform = transforms.Compose([
                                             transforms.Resize((224, 224)),
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
        self.split = flags.split

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
        self.img_path = './data/PACS/kfold'
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
        self.split = flags.split

    def loader(self, domain):
        file = './data/PACS/txt/{}.txt'.format(domain)
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
        test_file = './data/PACS/txt/{}.txt'.format(self.target)
        test = _Dataset(test_file, self.img_path)
        test_loader = Data.DataLoader(test, batch_size=self.test_batchsz, shuffle=True, num_workers=0)
        return test_loader

class VLCS_Loader():
    def __init__(self, flags):
        self.img_path = './data/VLCS/IMG'
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
        self.split = flags.split

    def loader(self, domain):
        file = '/data/VLCS/txt/{}_train.csv'.format(domain)
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
        test_file = './data/VLCS/txt/{}_test.csv'.format(self.target)
        test = _Dataset(test_file, self.img_path)
        test_loader = Data.DataLoader(test, batch_size=self.test_batchsz, shuffle=True, num_workers=0)
        return test_loader

    def val_loader(self):
        val1 = '/data/VLCS/txt/{}_val.csv'.format(self.source[0])
        val2 = '/data/VLCS/txt/{}_val.csv'.format(self.source[1])
        val3 = '/data/VLCS/txt/{}_val.csv'.format(self.source[2])
        val1_data = _Dataset(val1, self.img_path)
        val2_data = _Dataset(val2, self.img_path)
        val3_data = _Dataset(val3, self.img_path)
        val1_loader = Data.DataLoader(val1_data, batch_size=self.test_batchsz, shuffle=True, num_workers=0)
        val2_loader = Data.DataLoader(val2_data, batch_size=self.test_batchsz, shuffle=True, num_workers=0)
        val3_loader = Data.DataLoader(val3_data, batch_size=self.test_batchsz, shuffle=True, num_workers=0)
        return val1_loader, val2_loader, val3_loader


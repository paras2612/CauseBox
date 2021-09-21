import numpy as np
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler


class IHDPNPZDataset(torch.utils.data.Dataset):
    def __init__(self, np_data):
        self.mu1 = np_data['mu1'][:, 1]
        self.mu0 = np_data['mu0'][:, 1]
        self.t = np_data['t'][:, 1]
        self.x = np_data['x'][:, 1]
        self.yf = np_data['yf'][:, 1]
        self.ycf = np_data['ycf'][:, 1]
        self.len = self.x.shape[0]
        self.binary_indices = []
        self.continuous_indices = []
        for i, x in enumerate(self.x[1, :]):
            if x in (0, 1, 2):
                self.binary_indices.append(i)
            else:
                self.continuous_indices.append(i)

    def __getitem__(self, idx):
        return self.mu1[idx], self.mu0[idx], self.t[idx], self.x[idx], self.yf[idx], self.ycf[idx]

    def __len__(self):
        return self.len

    def indices_each_features(self):
        return self.binary_indices, self.continuous_indices


class IHDPNPZDataLoader(object):
    def __init__(self, train, test, cuda=False):
        self.cuda = cuda
        self.test_set = test
        self.train_set = train

    def train_loader(self, batch_size):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_set, batch_size=batch_size)

        return train_loader

    def test_loader(self, batch_size):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=self.cuda)

        return test_loader

    def loaders(self, batch_size):
        train_loader = self.train_loader(batch_size)
        test_loader = self.test_loader(batch_size)

        return train_loader, test_loader


class IHDPDataset(torch.utils.data.Dataset):
    def __init__(self, data):

        self.length = data.shape[0]
        self.t = data[:, 0]
        self.yf = data[:, 1]

        # Zero mean, unit variance for y during training
        self.y_mean, self.y_std = np.mean(self.yf), np.std(self.yf)
        self.standard_yf = (self.yf - self.y_mean) / self.y_std

        self.ycf = data[:, 2]
        self.mu0 = data[:, 3]
        self.mu1 = data[:, 4]
        self.x = data[:, 5:]

        self.x[:, 13] -= 1  # {1, 2} -> {0, 1}
        self.binary_indices = list(range(6, 25))
        self.continuous_indices = list(range(0, 6))

    def __getitem__(self, index):
        return self.mu1[index], self.mu0[index], self.t[index], self.x[index], self.yf[index], self.ycf[index], self.standard_yf[index]

    def __len__(self):
        return self.length

    def indices_each_features(self):
        return self.binary_indices, self.continuous_indices

    def y_mean_std(self):
        return self.y_mean, self.y_std


class IHDPDataLoader(object):
    def __init__(self, dataset, validation_split, shuffle=True):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)
        train_indices, valid_indices = indices[split:], indices[: split]

        self.dataset = dataset
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(valid_indices)

    def train_loader(self, batch_size):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=batch_size, sampler=self.train_sampler)

        return train_loader

    def test_loader(self, batch_size):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=batch_size, sampler=self.valid_sampler)

        return test_loader

    def loaders(self, batch_size):
        train_loader = self.train_loader(batch_size)
        test_loader = self.test_loader(batch_size)

        return train_loader, test_loader

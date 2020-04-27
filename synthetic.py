import scipy
from scipy.stats import powerlaw
import numpy as np
import math
import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticData():
    def __init__(self, num_of_worker, a, alpha, beta, seed=2020):
        self.alpha = alpha
        self.beta = beta
        self.num_of_workers = num_of_worker
        np.random.seed(seed)
        self.num_of_samples = [ math.ceil(i) for i in powerlaw.rvs(a, size=self.num_of_workers) * 2000 ]
        self.X = [ ]
        self.Y = [ ]

    def get_dataset(self):
        for i in range(self.num_of_workers):
            # np.random.seed(i)
            B = self.get_B()
            v_k = self.get_v(B)
            x = self.get_x(v_k, self.num_of_samples[ i ])
            u = self.get_u()
            W = self.get_W(u)
            b = self.get_b(u)
            y = self.get_y(W, x, b)
            self.X.append(x)
            self.Y.append(y)
        return self.X, self.Y

    def get_ratio(self, rank):
        # print(sum([ len(i) for i in self.X ]))
        return len(self.X[ rank ]) / sum([ len(i) for i in self.X ])

    def get_y(self, W, x, b):
        y = scipy.special.softmax(np.dot(W.T, x.T).T + b, axis=1)
        return np.array([ np.where(i == np.amax(i)) for i in y ]).reshape(-1)

    def get_W(self, u_k):
        return np.random.normal(loc=u_k, scale=1, size=(60, 10))

    def get_b(self, u_k):
        return np.random.normal(loc=u_k, scale=1)

    def get_u(self):
        return np.random.normal(loc=0.0, scale=self.alpha, size=10)

    def get_x(self, v_k, n):
        j = [ i ** -1.2 if i != 0 else 0 for i in range(60) ]
        return np.random.normal(loc=v_k, scale=j, size=(n, 60))

    def get_v(self, B):
        return np.random.normal(loc=B, scale=1, size=60)

    def get_B(self):
        return np.random.normal(loc=0.0, scale=self.beta, size=1)


class Synthetic_Dataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        assert len(x) == len(y)
        self._x = x
        self._y = y

    def __len__(self):
        return len(self._x)

    def __getitem__(self, index):
        x_item = torch.tensor(self._x[ index ], dtype=torch.float)
        y_item = torch.tensor(self._y[ index ])
        return x_item, y_item


def partition_dataset(rank, num_of_worker, a, alpha, beta, args):
    a = SyntheticData(num_of_worker, a, alpha, beta)
    x, y = a.get_dataset()
    x1, y1 = x[ rank ], y[ rank ]
    ratio = a.get_ratio(rank)
    x2 = np.concatenate(x)
    y2 = np.concatenate(y)
    myDataset = Synthetic_Dataset(x1, y1)
    testDataset = Synthetic_Dataset(x2, y2)
    train_loader = DataLoader(myDataset,
                              batch_size=args.bs,
                              shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(testDataset,
                              batch_size=args.bs,
                              shuffle=True,
                              pin_memory=True)
    return train_loader, test_loader, ratio, x, y
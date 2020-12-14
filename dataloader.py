from __future__ import print_function
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from torch.utils.data import Dataset
import ipdb
import numpy as np

class ECGDataset(Dataset):

    def __init__(self, data_dict):
        ### BEGIN SOLUTION
        self.X, self.Y, self.K_beat, self.K_thythm, self.K_freq = data_dict['X'], data_dict['Y'], data_dict['K_beat'], data_dict['K_rhythm'], data_dict['K_freq']
        ### END SOLUTION

    def __len__(self):
        """
        TODO: Denotes the total number of samples
        """

        ### BEGIN SOLUTION
        return len(self.Y)
        ### END SOLUTION

    def __getitem__(self, index):
        """
        TODO: Generates one sample of data
        """

        ### BEGIN SOLUTION
        return (self.X[:, index, :],
                self.K_beat[:, index, :],
                self.K_thythm[:, index, :],
                self.K_freq[:, index, :]), self.Y[index]
        return (torch.tensor(self.X[:, index, :]),
                torch.tensor(self.K_beat[:, index, :]),
                torch.tensor(self.K_thythm[:, index, :]),
                torch.tensor(self.K_freq[:, index, :])), torch.tensor(self.Y[index])
        ### END SOLUTION


from torch.utils.data import DataLoader


def load_data(dataset, batch_size=128):
    """
    TODO: load the `ECGDataset` it to dataloader. Set batchsize to 32.
    """
    ### BEGIN SOLUTION
    def my_collate(batch):
        X = torch.tensor([[_x[0][0][c] for _x in batch] for c in range(4)], dtype=torch.float)
        K_beat = torch.tensor([[_x[0][1][c] for _x in batch] for c in range(4)], dtype=torch.float)
        K_rhythm = torch.tensor([[_x[0][2][c] for _x in batch] for c in range(4)], dtype=torch.float)
        K_freq = torch.tensor([[_x[0][3][c] for _x in batch] for c in range(4)], dtype=torch.float)
        Y = torch.tensor([_x[1] for _x in batch], dtype=torch.long)
        return (X, K_beat, K_rhythm, K_freq), Y
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate)
    ### END SOLUTION


def get_dataloader(data_path=r'G:\MINA\data\challenge2017\1000_cached_data_permuted7', which='train', batch_size=128):
    dataset = ECGDataset(pd.read_pickle(os.path.join(data_path, '%s.pkl'%which)))
    return load_data(dataset, batch_size=batch_size)

from __future__ import print_function
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
from torch.utils.data import Dataset

class ECGDataset(Dataset):

    def __init__(self, data_dict):
        """
        TODO: init the Dataset instance.
        """
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

    def __getitem__(self, i):
        """
        TODO: Generates one sample of data
            return the ((X, K_beat, K_rhythm, K_freq), Y) for the i-th data.
            Be careful about which dimension you are indexing.
        """

        ### BEGIN SOLUTION
        return (self.X[:, i, :], self.K_beat[:, i, :], self.K_thythm[:, i, :], self.K_freq[:, i, :]), self.Y[i]
        ### END SOLUTION

def load_data(dataset, batch_size=128):
    """
    Return a DataLoader instance basing on a Dataset instance, with batch_size specified.
    Note that since the data has already been shuffled, we set shuffle=False
    """
    def my_collate(batch):
        """
        :param batch: this is essentially [dataset[i] for i in [...]]
        batch[i] should be ((Xi, Ki_beat, Ki_rhythm, Ki_freq), Yi)
        TODO: write a collate function such that it outputs ((X, K_beat, K_rhythm, K_freq), Y)
            each output variable is a batched version of what's in the input *batch*
            For each output variable - it should be either float tensor or long tensor (for Y). If applicable, channel dim precedes batch dim
            e.g. the shape of each Xi is (# channels, n). In the output, X should be of shape (batch_size, # channels, n)
        """
        ### BEGIN SOLUTION
        X = torch.tensor([[_x[0][0][c] for _x in batch] for c in range(4)], dtype=torch.float)
        K_beat = torch.tensor([[_x[0][1][c] for _x in batch] for c in range(4)], dtype=torch.float)
        K_rhythm = torch.tensor([[_x[0][2][c] for _x in batch] for c in range(4)], dtype=torch.float)
        K_freq = torch.tensor([[_x[0][3][c] for _x in batch] for c in range(4)], dtype=torch.float)
        Y = torch.tensor([_x[1] for _x in batch], dtype=torch.long)
        ### END SOLUTION
        return (X, K_beat, K_rhythm, K_freq), Y

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate)


def main(data_path=None):
    data_path = r'G:\MINA\data\challenge2017\100_cached_data_permuted7'
    train_dict = pd.read_pickle(os.path.join(data_path, 'train.pkl'))
    test_dict = pd.read_pickle(os.path.join(data_path, 'test.pkl'))
    print(f"There are {len(train_dict['Y'])} training data, {len(test_dict['Y'])} test data")
    print(f"Shape of X: {train_dict['X'][:, 0,:].shape} = (#channels, n)")
    print(f"Shape of beat feature: {train_dict['K_beat'][:, 0, :].shape} = (#channels, n)")
    print(f"Shape of rhythm feature: {train_dict['K_rhythm'][:, 0, :].shape} = (#channels, M)")
    print(f"Shape of frequency feature: {train_dict['K_freq'][:, 0, :].shape} = (#channels, 1)")

    train_loader = load_data(ECGDataset(train_dict))
    test_loader = load_data(ECGDataset(test_dict))

    #=========test:
    assert len(train_loader.dataset) == 1696, "Length of training data incorrect."
    assert len(test_loader.dataset) == 425, "Length of test data incorrect."
    assert len(train_loader) == 14, "Length of the training dataloader incorrect - maybe check batch_size"
    assert len(test_loader) == 4, "Length of the testing dataloader incorrect - maybe check batch_size"
    assert [x.shape for x in train_loader.dataset[0][0]] == [(4,3000), (4,3000), (4,60), (4,1)], "Shapes of the data don't match. Check __getitem__ implementation"
    assert [0, 1, 0, 0, 1, 0, 0, 1, 0, 1] == [test_loader.dataset[i][1] for i in range(10)], "Data seems permuted. Shuffle should be set to False"


    return train_loader, test_loader

if __name__=='__main__':
    main()
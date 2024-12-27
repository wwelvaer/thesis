import h5torch
import numpy as np
import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence

VOCABULARY = {
    'B': 0,'2': 1,'%': 2, 's': 3, '@': 4, '0': 5, '4': 6, 'a': 7, ' ': 8,
    'I': 9, 'n': 10, 'P': 11, 'c': 12, '1': 13, '9': 14, '[': 15, 'C': 16,
    'h': 17, 'o': 18, 'N': 19, '\\': 20, 't': 21, '6': 22, 'e': 23, 'F': 24,
    'r': 25, 'O': 26, '8': 27, 'S': 28, '3': 29, '/': 30, '5': 31, '+': 32,
    '=': 33, '#': 34, 'A': 35, '(': 36, '.': 37, ']': 38, 'H': 39, ')': 40,
    '-': 41, 'i': 42, '7': 43, 'l': 44
}

def bin_spectrum(mzs, intensities):
    return (
        np.histogram(mzs, np.arange(10, 1_000 + 1e-8, 1), weights=intensities)[0]
    )


class GNPSDataModule(LightningDataModule):
    def __init__(
        self,
        path,
        batch_size=16,  # batch size for model
        n_workers=4,  # num workers in dataloader
        in_memory=True,  # whether to use h5torch in-memory mode for more-efficient dataloading
    ):
        super().__init__()

        self.batch_size = batch_size
        self.n_workers = n_workers
        self.path = path
        self.in_memory = in_memory

    def setup(self, stage):
        train_indices, val_indices, test_indices = np.split(np.arange(56105), [45000, 50000])

        self.train = h5torch.Dataset(
            self.path,
            sample_processor=self.sample_processor,
            in_memory=self.in_memory,
            subset=train_indices,
        )

        self.val = h5torch.Dataset(
            self.path,
            sample_processor=self.sample_processor,
            in_memory=self.in_memory,
            subset=val_indices,
        )

        self.test = h5torch.Dataset(
            self.path,
            sample_processor=self.sample_processor,
            in_memory=self.in_memory,
            subset=test_indices,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=batch_collater,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            collate_fn=batch_collater,
        )

    @staticmethod
    def sample_processor(f, sample):
        return {
            "fingerprint" : sample['0/fingerprint'],
            "spectrum" : bin_spectrum(sample['0/mzs'], sample['0/intensities']),
            "smiles" : np.array([VOCABULARY[l] for l in sample["central"].astype(str)])
        }


def batch_collater(batch):
    batch_collated = {}
    keys = list(batch[0])
    for k in keys:
        v = [b[k] for b in batch]
        if len({t.shape for t in v}) == 1:
            batch_collated[k] = torch.tensor(np.array(v))
        else:
            batch_collated[k] = pad_sequence(
                [torch.tensor(t) for t in v], batch_first=True, padding_value=-1
            )
    return batch_collated
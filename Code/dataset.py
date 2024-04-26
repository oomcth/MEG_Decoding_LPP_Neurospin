import torch
import typing as tp
from os import listdir
from os.path import isfile, join
import random
from tqdm import tqdm
import torch.utils.data.dataloader as dataloader
from torch.utils.data import ConcatDataset
import numpy as np


path = '/Volumes/KINGSTON/Train_DATA_MATHIS/'
files = [f for f in listdir(path) if isfile(join(path, f))]
files = list(map(lambda x: path + x, files))
print("file count:", len(files))


class Segment_Batch():
    def __init__(self, batch_meg: torch.tensor,
                 batch_phoneme: torch.tensor,
                 batch_subject: torch.tensor) -> None:
        self.batch_meg = batch_meg
        self.batch_phoneme = batch_phoneme
        self.batch_subject = batch_subject

    def to(self, device: tp.Any):
        self.batch_meg = self.batch_meg.to(device)
        self.batch_phoneme = self.batch_phoneme.to(device)
        self.batch_subject = self.batch_subject.to(device)
        return self

    def __getitem__(self, index: tp.Any):
        return self.batch_meg[index]

    def size(self):
        return self.batch_meg.size()

    def __len__(self) -> int:
        return self.batch_meg.size(0)

    def cat(self, other_seg_batch):
        self.batch_meg = torch.cat([self.batch_meg,
                                   other_seg_batch.batch_meg])
        self.batch_phoneme = torch.cat([self.batch_phoneme,
                                       other_seg_batch.batch_phoneme])
        self.batch_subject = torch.cat([self.batch_subject,
                                       other_seg_batch.batch_subject])
        return self


class CustomDataset(dataloader.Dataset):
    def __init__(self, files):
        self.segment_batch = Segment_Batch(torch.Tensor(),
                                           torch.Tensor(),
                                           torch.Tensor())
        for file in files:
            segment_batch = torch.load(file)
            self.segment_batch.cat(segment_batch)
        del segment_batch
        self.segment_batch.batch_subject.to(torch.int64)

    def __len__(self):
        return self.segment_batch.batch_subject.size(0)

    def __getitem__(self, idx):
        return (self.segment_batch.batch_meg[idx],
                self.segment_batch.batch_phoneme[idx],
                self.segment_batch.batch_subject[idx])


def chunk_list(lst, m):
    return np.array_split(lst, (len(lst) + m - 1) // m)


def Create_Loader(files: list[str] = files,
                  batch_size: int = 512,
                  m: int = 5) -> dataloader.DataLoader:
    files = chunk_list(files, m=5)
    dataset = None
    for chunk in tqdm(files, desc="loading data", leave=False):
        if dataset is None:
            dataset = CustomDataset(chunk)
        else:
            dataset = ConcatDataset(
                [dataset, CustomDataset(chunk)]
            )
    loader = dataloader.DataLoader(dataset, batch_size=batch_size,
                                   shuffle=True)
    return loader


def split_indices(lst):
    indices = list(range(len(lst)))
    random.shuffle(indices)
    mid = len(indices) // 2
    return indices[:mid], indices[mid:]


def mix(path1: str, path2: str):
    data1 = torch.load(path1)
    data2 = torch.load(path2)
    merged_meg = torch.cat([data1.batch_meg, data2.batch_meg])
    merged_phoneme = torch.cat([data1.batch_phoneme, data2.batch_phoneme])
    merged_subject = torch.cat([data1.batch_subject, data2.batch_subject])
    i, j = split_indices(list(range(merged_meg.size(0))))
    data1 = Segment_Batch(merged_meg[i],
                          merged_phoneme[i],
                          merged_subject[i])
    data2 = Segment_Batch(merged_meg[j],
                          merged_phoneme[j],
                          merged_subject[j])
    torch.save(data1, path1)
    torch.save(data2, path2)


if __name__ == "__main__":
    for _ in tqdm(range(10_000), desc='merging'):
        path1, path2 = random.sample(files, 2)
        mix(path1, path2)

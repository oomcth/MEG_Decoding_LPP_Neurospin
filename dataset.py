import torch
import typing as tp
import mne


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

    def __getitem__(self, index):
        return self.batch_meg[index]

    def size(self):
        return self.batch_meg.size()

    def __len__(self) -> int:
        return self.batch_meg.size(0)


def mix(segA: Segment_Batch, segB: Segment_Batch):
    pass

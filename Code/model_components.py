import torch
import typing as tp
from torch import nn
from dataset import Segment_Batch
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, spatial_dim: int) -> None:
        super(SpatialAttention, self).__init__()
        self.query = torch.randn(spatial_dim, spatial_dim, requires_grad=True)
        self.key = torch.randn(spatial_dim, spatial_dim, requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, meg: torch.tensor) -> torch.tensor:
        batch_size, _, _ = meg.size()

        key = torch.bmm(
            self.key.expand(batch_size, -1, -1), meg
        ).permute(0, 2, 1)
        query = torch.bmm(self.query.expand(batch_size, -1, -1), meg)

        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=2)
        out = torch.bmm(attention, meg)
        return self.gamma * out + meg


class Subject_Layer(nn.Module):
    def __init__(self, n_subjects: int, in_channels: int,
                 out_channels: int) -> None:
        super(Subject_Layer, self).__init__()
        self.weights = nn.Parameter(
            torch.randn(n_subjects, in_channels, out_channels)
        )

    def forward(self, x: torch.tensor,
                subjects: torch.tensor) -> torch.tensor:
        _, C, D = self.weights.size()
        weights = self.weights.gather(
            0, subjects.view(-1, 1, 1).expand(-1, C, D)
        )
        return torch.einsum("bct,bcd->bdt", x, weights)


class LSTM_Layer(nn.Module):
    def __init__(self, dim: int, seq_len: int, norm: bool = True) -> None:
        super(LSTM_Layer, self).__init__()
        self.lstm = nn.LSTM(seq_len, seq_len, 1, batch_first=True)
        self.hidden = (torch.zeros([1, 1, seq_len]),
                       torch.zeros([1, 1, seq_len]))
        self.norm = norm
        if norm:
            self.normo = nn.LayerNorm([dim, seq_len])
            self.normh = nn.LayerNorm([seq_len])

    def forward(self, x: torch.tensor, hidden: tp.Any = None):
        if hidden is None:
            (output, hidden) = self.lstm(
                x, ((self.hidden[0].expand(-1, x.size(0), -1)),
                    (self.hidden[1].expand(-1, x.size(0), -1))))
        else:
            (output, hidden) = self.lstm(x, hidden)
        if self.norm:
            output = self.normo(output)
            hidden = (self.normh(hidden[0]),
                      self.normh(hidden[1]))
        return output, (hidden[0], hidden[1])


class model(nn.Module):
    def __init__(self, spatial_dim: int, temporal_dim: int, n_subjects: int,
                 num_class: int, dropout: float = 0.2) -> None:
        super(model, self).__init__()
        self.attention_layer = SpatialAttention(spatial_dim)
        self.subject_layer = Subject_Layer(n_subjects, spatial_dim,
                                           spatial_dim)
        self.lstm1 = LSTM_Layer(spatial_dim, temporal_dim, True)
        self.lstm2 = LSTM_Layer(spatial_dim, temporal_dim, True)
        self.lstm3 = LSTM_Layer(spatial_dim, temporal_dim, True)
        self.classifier = nn.Linear(temporal_dim, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seg_batch: Segment_Batch):
        x, subjects = seg_batch.batch_meg, seg_batch.batch_subject
        x = self.attention_layer(x)
        x = self.subject_layer(x, subjects)
        x = self.lstm1(x)
        x = self.lstm2(*x)
        out, _ = self.lstm3(*x)
        out = self.dropout(out[:, -1, :])
        return self.classifier(out)

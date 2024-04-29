import torch
import typing as tp
from torch import nn
from dataset import Segment_Batch


@torch.jit.script
def swiglu(x):
    return x * torch.sigmoid(x) * (1 + torch.sigmoid(x))


class SpatialAttention(nn.Module):
    def __init__(self, device='cuda') -> None:
        super(SpatialAttention, self).__init__()
        self.Ut = torch.tensor(torch.load("/Volumes/KINGSTON/Graph/Udef.pth"),
                               dtype=torch.float, device=device)
        self.Ut = self.Ut.T

    def forward(self, meg: torch.tensor) -> torch.tensor:
        out = torch.bmm(self.Ut.expand(meg.size(0), -1, -1), meg)
        return out


class Subject_Layer(nn.Module):
    def __init__(self, n_subjects: int, in_channels: int,
                 out_channels: int) -> None:
        super(Subject_Layer, self).__init__()
        self.weights = nn.Parameter(
            torch.randn(n_subjects, in_channels, out_channels),
            requires_grad=True
        )
        self.zero = torch.tensor(0, dtype=torch.int64)

    def forward(self, x: torch.tensor,
                subjects: torch.tensor) -> torch.tensor:
        _, C, D = self.weights.size()
        weights = self.weights.gather(
            self.zero, subjects.view(-1, 1, 1).expand(-1, C, D).to(torch.int64)
        )
        return torch.einsum("bct,bcd->bdt", x, weights)


class LSTM_Layer(nn.Module):
    def __init__(self, dim: int, seq_len: int, norm: bool = True,
                 device='cuda') -> None:
        super(LSTM_Layer, self).__init__()
        self.lstm = nn.LSTM(seq_len, seq_len, 1, batch_first=True)
        self.hidden = (torch.zeros([1, 1, seq_len], device=device),
                       torch.zeros([1, 1, seq_len], device=device))
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
                 num_class: int, dropout: float = 0.2, device='cuda',
                 lstm_norm=True) -> None:
        super(model, self).__init__()
        self.attention_layer = SpatialAttention(device)
        self.subject_layer = Subject_Layer(n_subjects, spatial_dim,
                                           spatial_dim)
        self.lstm1 = LSTM_Layer(spatial_dim, temporal_dim, lstm_norm, device)
        self.lstm2 = LSTM_Layer(spatial_dim, temporal_dim, lstm_norm, device)
        self.lstm3 = LSTM_Layer(spatial_dim, temporal_dim, lstm_norm, device)
        self.classifier = nn.Linear(temporal_dim, num_class)
        self.dropout = nn.Dropout(dropout)
        self.params = nn.Parameter(
            torch.zeros(2, device=device), requires_grad=True
        )

    def forward(self, seg_batch: Segment_Batch):
        x, subjects = seg_batch.batch_meg, seg_batch.batch_subject
        x = self.attention_layer(x)
        x = self.dropout(x)
        x = self.subject_layer(x, subjects)
        initial_x = x.clone()
        x = self.lstm1(x)
        x = (x[0] + self.params[0] * initial_x,
             x[1])
        x = self.lstm2(*x)
        x = (x[0] + self.params[1] * initial_x,
             x[1])
        out, _ = self.lstm3(*x)
        out = swiglu(out[:, -1, :])
        return self.classifier(out)

import torch.nn as nn


class Classification_Loss(nn.Module):
    def __init__(self, num_classes=10):
        super(Classification_Loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        return self.loss(x, y)

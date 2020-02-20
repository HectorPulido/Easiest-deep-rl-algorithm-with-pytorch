import torch
from torch import nn
from torch.nn import functional as F

class MlpAgent(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MlpAgent, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, H)
        self.linear3 = nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        y_pred = self.linear3(y_pred)
        return F.softmax(y_pred, dim=-1)
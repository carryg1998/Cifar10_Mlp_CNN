import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, hidden_num):
        super(Mlp, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_channel, hidden_channel, bias=True), nn.ReLU())
        self.hidden_num = hidden_num
        if hidden_num >= 2:
            self.hidden = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_channel, hidden_channel, bias=True), nn.ReLU())
                for i in range(hidden_num-1)])
        self.out_layer = nn.Linear(hidden_channel, out_channel, bias=True)

    def forward(self, x):
        x = self.layer1(x)
        if self.hidden_num >= 2:
            for blk in self.hidden:
                x = blk(x)
        out = nn.functional.softmax(self.out_layer(x), dim=1)
        return out
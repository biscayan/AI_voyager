import torch
import timm
from torch import nn

class CompareCoAt(nn.Module):
    def __init__(self):
        super(CompareCoAt, self).__init__()
        self.coat = timm.create_model('coat_mini', pretrained=True)
        self.fc_layer = nn.Linear(1000, 1)

    def forward(self, input):
        x = self.coat(input)
        output = self.fc_layer(x)
        return output


class CompareNet(nn.Module):
    def __init__(self):
        super(CompareNet, self).__init__()
        self.before_net = CompareCoAt()
        self.after_net = CompareCoAt()

    def forward(self, before_input, after_input):
        before = self.before_net(before_input)
        after = self.after_net(after_input)
        delta = after - before
        return delta
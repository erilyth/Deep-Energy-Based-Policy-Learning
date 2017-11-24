import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        # State 2D + Noise 2D
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)
        init.normal(self.fc1.weight, mean=0, std=0.5)
        init.normal(self.fc2.weight, mean=0, std=0.5)
        init.normal(self.fc3.weight, mean=0, std=0.5)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

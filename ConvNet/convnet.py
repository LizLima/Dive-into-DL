import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(6)
        self.avg1  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.bn2   = nn.BatchNorm2d(16)
        self.avg2  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(in_features=400, out_features=120)
        self.fc2   = nn.Linear(in_features=120, out_features=84)
        self.fc3   = nn.Linear(in_features=84, out_features=10)

        self.relu = nn.LeakyReLU(negative_slope=0.01)
    def forward(self, input):
        x = self.relu(self.bn1(self.conv1(input)))
        x = self.avg1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avg2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    @staticmethod
    def init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        if classname.find('Conv2d') != -1:
            m.weight.data.normal_(0.0, 0.02)

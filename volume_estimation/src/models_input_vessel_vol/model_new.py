import torch.nn as nn
import torch

# from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np


# Define the neural network architecture
class VolumeNet(nn.Module):
    def __init__(self):
        super(VolumeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, padding=2
        )
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=5, padding=2
        )

        self.bn4 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(
            in_features=64 * 520, out_features=1024
        )  # 520 = 20*26, otherwise 4800
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=1)

    def forward(self, vessel_depth, liquid_depth, vessel_vol):
        # print(depth_image.shape)
        # x = depth_image.unsqueeze(1)  # add channel dimension
        # print(depth_image.shape)
        x = torch.stack(
            [vessel_depth, liquid_depth, vessel_vol], dim=1
        )  # stack the two input tensors along the channel dimension
        # print(x.shape)

        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.batch_norm(x, momentum=0.1, eps=1e-5)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.batch_norm(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn4(self.conv4(x)))
        # dropout layer
        x = F.dropout2d(x, p=0.2)
        # print(x.shape)

        x = x.view(-1, 64 * 520)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Network Training script that mainly utilizes PyTorch
# Incomplete Script

import torch
import torch.nn as nn
import torch.nn.functional as functional

class Net(nn.Module):   # Define neural network
    def __init__(self):
        super(Net, self).__init__()
        # 3 input channels, 1 output channel, 256x256 square convolution
        self.conv1 = nn.Conv2d(3, 1, 256)
        self.conv2 = nn.Conv2d(1, 3, 256)

        


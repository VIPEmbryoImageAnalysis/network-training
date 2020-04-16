# Network Training script that mainly utilizes PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as functional

class Net(nn.Module):   # Define neural network
    def __init__(self):
        super(Net, self).__init__()
        # Incomplete

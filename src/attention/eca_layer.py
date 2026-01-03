import torch
import torch.nn as nn
import math

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # adaptive kernel size
        t = int(abs((math.log2(channels) / gamma) + b))
        k = t if t % 2 else t + 1  

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            padding=k // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                
        y = y.squeeze(-1).squeeze(-1)       
        y = y.unsqueeze(1)                   
        y = self.conv(y)                      
        y = self.sigmoid(y)                  
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1) 
        return x * y

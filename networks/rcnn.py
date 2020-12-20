import torch
import torch.nn as nn

from networks.layers import (
    RecurrentConvolutionalLayer,
    ConvLayer,
    Dense
)

class SharkRCNN(nn.Module):
    def __init__(self, num_steps):
        super(SharkRCNN, self).__init__()
        
        self.conv = ConvLayer(1, 32, (7,3), (2,1), (3,1))
        self.pool1 = nn.MaxPool2d((2,1),(2,1))
        
        self.dropout1 = nn.Dropout(.5)
        
        self.rcl1 = RecurrentConvolutionalLayer(32, 64, (3,2), (1,1), num_steps)
        self.dropout2 = nn.Dropout(.4)
        
        self.rcl2 = RecurrentConvolutionalLayer(64, 128, (5,2), (2,1), num_steps)
        self.dropout3 = nn.Dropout(.3)
        
        self.pool2 = nn.MaxPool2d((2,1),(2,1))
        
        self.rcl3 = RecurrentConvolutionalLayer(128, 256, (7,1), (3,0), num_steps)
        self.dropout4 = nn.Dropout(.2)
        
        self.rcl4 = RecurrentConvolutionalLayer(256, 512, (1,1), (0,0), num_steps)
        self.gap_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc = Dense([512, 256, 64, 4])
        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool1(x)
        x = self.dropout1(x)
       
        x = self.rcl1(x)
        x = self.dropout2(x)
        
        x = self.rcl2(x)
        x = self.dropout3(x)
        
        x = self.pool2(x)
        
        x = self.rcl3(x)
        x = self.dropout4(x)
        
        x = self.rcl4(x)
        x = self.gap_pool(x)
        
        x = self.fc(x)
        
        return x
    
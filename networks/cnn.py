import torch
import torch.nn as nn
from networks.layers import (
    ConvLayer,
    Dense,
    NonLocalBlock
)

class SharkVGG(nn.Module):
    def __init__(self, chn, dim='2d', non_local=False):
        super(SharkVGG, self).__init__()
        chan = 32

        temporal = dim == '1d'

        if temporal:
            dim = 1
            fc_dim = 1536

            ks = [7, 5, 3]
            pad = [3, 2, 1]

            pool = nn.MaxPool1d
        else:
            dim = 2
            fc_dim = 6144
            # ks = [(3,2), (5,2), (7,1)]
            # pad = [(1,1), (2,1), (3,0)]

            ks = [(7,1), (5,2), (3,2)]
            pad = [(3,0), (2,1),(1,1)]

            pool = nn.MaxPool2d

        self.non_local = non_local
        
        self.conv1 = nn.Sequential(
            ConvLayer(chn, chan, ks[0], 1, pad[0], temporal=temporal),
            ConvLayer(chan, chan, ks[0], 1, pad[0], temporal=temporal),
            pool(2),
            nn.Dropout(.5)
            )

        self.conv2 = nn.Sequential(
            ConvLayer(chan, chan*2, ks[1], 1, pad[1], temporal=temporal),
            ConvLayer(chan*2, chan*2, ks[1], 1, pad[1], temporal=temporal),
            pool(2),
            nn.Dropout(.5)
            )

        self.conv3 = nn.Sequential(
            ConvLayer(chan*2, chan*4, ks[2], 1, pad[2], temporal=temporal),
            ConvLayer(chan*4, chan*4, ks[2], 1, pad[2], temporal=temporal)
            )

        if self.non_local:
            self.nlb = NonLocalBlock(chan*2, dim=dim)

        self.fc = Dense([fc_dim, 1024, 128, 4])
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        if self.non_local:
            x = self.nlb(x)

        x = self.conv3(x)

        # print(x.shape)
        
        x = self.fc(x)
        return x

'''
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def conv_layer(chanIn, chanOut, kernel_size = 3, padding=0, temporal=False):
    if temporal:
        layer = nn.Sequential(
            nn.Conv1d(chanIn, chanOut, kernel_size, padding=padding),        
            nn.ReLU(),
            nn.BatchNorm1d(chanOut)
            )
    else:
        layer = nn.Sequential(
            nn.Conv2d(chanIn, chanOut, kernel_size, padding=padding),        
            nn.ReLU(),
            nn.BatchNorm2d(chanOut)
            )
        
    return layer

def dense(chanIn):
    return nn.Sequential(
        Flatten(),
        nn.Dropout(.25),
        nn.ReLU(),
        nn.Linear(chanIn, 1000),
        nn.ReLU(),
        nn.BatchNorm1d(1000),
        nn.Linear(1000,100),
        nn.ReLU(),
        nn.BatchNorm1d(100),
        nn.Dropout(.25),
        nn.Linear(100, 4)
    )

class SharkVGG(nn.Module):
    def __init__(self, chn, non_local=False):
        super(SharkVGG, self).__init__()
        chan = 32
        ks = 5
        pad = ks//2

        self.non_local = non_local
        
        self.fc = dense(768*2)
        
        self.conv11 = conv_layer(chn, chan, kernel_size=ks, padding=pad, temporal=True)
        self.conv12 = conv_layer(chan, chan, kernel_size=ks, padding=pad, temporal=True)
        self.pool12 = nn.MaxPool1d(2,2)
        self.dp_1 = nn.Dropout(.5)
        self.conv13 = conv_layer(chan, chan*2, kernel_size=ks, padding=pad, temporal=True)
        self.conv14 = conv_layer(chan*2, chan*2, kernel_size=ks, padding=pad, temporal=True)
        self.pool15 = nn.MaxPool1d(2,2)
        self.dp_2 = nn.Dropout(.5)
        self.conv16 = conv_layer(chan*2, chan*4, kernel_size=ks, padding=pad, temporal=True)
        self.conv17 = conv_layer(chan*4, chan*4, kernel_size=ks, padding=pad, temporal=True)

        if self.non_local:
            self.nlb1 = NonLocalBlock(chan, dim=1)
            self.nlb2 = NonLocalBlock(chan*2, dim=1)
            self.nlb3 = NonLocalBlock(chan*4, dim=1)
        
        
    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.pool12(x)
        x = self.dp_1(x)

        if self.non_local:
            x = self.nlb1(x)

        x = self.conv13(x)
        x = self.conv14(x)
        x = self.pool15(x)
        x = self.dp_2(x)

        if self.non_local:
            x = self.nlb2(x)
        
        x = self.conv16(x)
        x = self.conv17(x)

        if self.non_local:
            x = self.nlb3(x)
        
        x = self.fc(x)
        return x

class SharkVGG2d(nn.Module):
    def __init__(self, chn, non_local=False):
        super(SharkVGG2d, self).__init__()
        chan = 32
        ks = [(3,2), (5,2), (7,1)]
        pad = [(1,1), (2,1), (3,0)]

        self.non_local = non_local
        
        self.fc = dense(4608)
        #self.fc = dense(3072)
        
        self.conv11 = conv_layer(chn, chan, kernel_size=ks[0], padding=pad[0])
        self.conv12 = conv_layer(chan, chan, kernel_size=ks[0], padding=pad[0])
        self.pool1 = nn.MaxPool2d(2,2)
        self.dp_1 = nn.Dropout(.5)
        self.conv13 = conv_layer(chan, chan*2, kernel_size=ks[1], padding=pad[1])
        self.conv14 = conv_layer(chan*2, chan*2, kernel_size=ks[1], padding=pad[1])
        self.pool2 = nn.MaxPool2d(2,2)
        self.dp_2 = nn.Dropout(.5)
        self.conv16 = conv_layer(chan*2, chan*4, kernel_size=ks[2], padding=pad[2])
        self.conv17 = conv_layer(chan*4, chan*4, kernel_size=ks[2], padding=pad[2])

        if self.non_local:
            self.nlb1 = NonLocalBlock(chan, dim=2)
            self.nlb2 = NonLocalBlock(chan*2, dim=2)
            self.nlb3 = NonLocalBlock(chan*4, dim=2)
        
        
    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.pool1(x)
        x = self.dp_1(x)

        if self.non_local:
            x = self.nlb1(x)

        x = self.conv13(x)
        x = self.conv14(x)
        x = self.pool2(x)
        x = self.dp_2(x)

        if self.non_local:
            x = self.nlb2(x)
        
        x = self.conv16(x)
        x = self.conv17(x)

        if self.non_local:
            x = self.nlb3(x)
        
        x = self.fc(x)
        return x
'''

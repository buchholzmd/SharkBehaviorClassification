import torch
import torch.nn as nn

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
#         nn.ReLU(),
        nn.Linear(chanIn, 1000),
        nn.ReLU(),
        nn.BatchNorm1d(1000),
        nn.Linear(1000,100),
        nn.ReLU(),
        nn.BatchNorm1d(100),
#         nn.Dropout(.25),
        nn.Linear(100, 4)
    )


class SharkVGG(nn.Module):
    def __init__(self, chn):
        super(SharkVGG, self).__init__()
        chan = 32
        ks = 5
        pad = ks//2
        
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
        
        
    def forward(self, x):
        # print('in', x.shape)
        x = self.conv11(x)
        x = self.conv12(x)
        # print('first', x.shape)
        
        x = self.pool12(x)
        x = self.dp_1(x)
        x = self.conv13(x)
        x = self.conv14(x)
        # print('second', x.shape)
        
        x = self.pool15(x)
        x = self.dp_2(x)
        
        x = self.conv16(x)
        x = self.conv17(x)
        # print('thrid', x.shape)
        
        x = self.fc(x)
        return x

class SharkVGG2d(nn.Module):
    def __init__(self, chn):
        super(SharkVGG2d, self).__init__()
        chan = 32
        ks = [(3,2), (5,2), (7,1)]
        pad = [(1,1), (2,1), (3,0)]
        
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
        
        
    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        
        x = self.pool1(x)
        x = self.dp_1(x)
        x = self.conv13(x)
        x = self.conv14(x)
        
        x = self.pool2(x)
        x = self.dp_2(x)
        
        x = self.conv16(x)
        x = self.conv17(x)
        
        x = self.fc(x)
        return x
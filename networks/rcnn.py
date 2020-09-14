import torch
import torch.nn as nn

class RecurrentConvolutionalLayer(nn.Module):
    def __init__(self, chanIn, chanOut, ks, pad, n):
        super(RecurrentConvolutionalLayer, self).__init__()
        self.conv1 = conv_layer(chanIn, chanOut, kernel_size=ks, padding=pad)
        self.rcl1 = conv_layer(chanIn, chanOut, kernel_size=ks, padding=pad)
        self.rcl2 = conv_layer(chanOut, chanOut, kernel_size=ks, padding=pad)
#         self.bn1 = nn.BatchNorm2d(chanIn)
#         self.bn2 = nn.BatchNorm2d(chanOut)
        
        self.lrn = nn.LocalResponseNorm(2)
        
        self.n = n
        
    def forward(self, x):
        x_in = self.conv1(x)
#         x = self.bn1(x)
        
        x = self.rcl1(x)
#         print("Layer " + str(1) + " rec: " + str(x.shape))
#         print("Layer " + str(1) + " inp: " + str(x_in.shape))
        x = x + x_in
        x = self.lrn(x)
        
        for i in range(1, self.n): 
            print(x.shape)
            x = self.rcl2(x)
            print(x.shape)
#             print("Layer " + str(i+1) + " rec: " + str(x.shape))
#             print("Layer " + str(i+1) + " inp: " + str(x_in.shape))
            x = x + x_in
            x = self.lrn(x)
#             x = self.bn2(x)
        
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def conv_layer(chanIn, chanOut, kernel_size=3, padding=0):
    return nn.Sequential(
        nn.Conv2d(chanIn, chanOut, kernel_size, padding=padding),        
        nn.ReLU()
        )

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


class SharkRCNN2d(nn.Module):
    def __init__(self, chn, num_steps):
#         super(SharkRCNN2d, self).__init__()
        super().__init__()
        chan = 32#8*2
        ks = [(2,3), (2,5), (1,7), (1,1)]
        pad = [(1,1), (1,2), (0,3), (0,0)]

        self.rcl1 = RecurrentConvolutionalLayer(chn, chan, ks[0], pad[0], num_steps)
        self.dropout1 = nn.Dropout(.5)
        self.rcl2 = RecurrentConvolutionalLayer(chan, chan*2, ks[1], pad[1], num_steps)
        self.pool1 = nn.MaxPool2d(2,2)
        self.dropout2 = nn.Dropout(.5)
        
        self.rcl3 = RecurrentConvolutionalLayer(chan*2, chan*4, ks[2], pad[2], num_steps)
        self.dropout3 = nn.Dropout(.5)
        self.rcl4 = RecurrentConvolutionalLayer(chan*4, chan*4, ks[3], pad[3], num_steps)
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.fc = dense(1536)
        
        
    def forward(self, x):
        x = self.rcl1(x)
        x = self.dropout1(x)
        x = self.rcl2(x)
        x = self.pool1(x)
        x = self.dropout2(x)
        x = self.rcl3(x)
        x = self.dropout3(x)
        x = self.rcl4(x)
        x = self.pool2(x)
        
        x = self.fc(x)
        return x
        
    
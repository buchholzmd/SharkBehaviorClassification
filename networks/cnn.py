import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def conv_layer(chanIn, chanOut, kernel_size = 3, padding=0):
    return nn.Sequential(
        nn.Conv1d(chanIn, chanOut, kernel_size, padding=padding),        
        nn.ReLU(),
        nn.BatchNorm1d(chanOut)
        )

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


class SharkVGG1(nn.Module):
    def __init__(self, chn):
        #super(SharkNet1, self).__init__()
        super().__init__()
        chan = 32#8*2
        ks = 5
        pad = ks//2
        
        self.fc = dense(768*2)
        
        self.conv11 = conv_layer(chn, chan, kernel_size=ks, padding=pad)
        self.conv12 = conv_layer(chan, chan, kernel_size=ks, padding=pad)
        self.pool12 = nn.MaxPool1d(2,2)
        self.dp_1 = nn.Dropout(.5)
        self.conv13 = conv_layer(chan, chan*2, kernel_size=ks, padding=pad)
        self.conv14 = conv_layer(chan*2, chan*2, kernel_size=ks, padding=pad)
        self.pool15 = nn.MaxPool1d(2,2)
        self.dp_2 = nn.Dropout(.5)
        self.conv16 = conv_layer(chan*2, chan*4, kernel_size=ks, padding=pad)
        self.conv17 = conv_layer(chan*4, chan*4, kernel_size=ks, padding=pad)
        
        
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
    
class SharkVGG2(nn.Module):
    def __init__(self, chn):
        super().__init__()
        chan = 32
        k = [3,5,7]
        
        self.fc = dense(4608)
        
        self.pool = nn.MaxPool1d(2,2)
        self.dropout = nn.Dropout(.5)
        
        self.conv11 = conv_layer(chn, chan, kernel_size=k[0], padding=k[0]//2)
        self.conv12 = conv_layer(chan, chan, kernel_size=k[0], padding=k[0]//2)
        self.conv13 = conv_layer(chan, chan*2, kernel_size=k[0], padding=k[0]//2)
        self.conv14 = conv_layer(chan*2, chan*2, kernel_size=k[0], padding=k[0]//2)
        self.conv15 = conv_layer(chan*2, chan*4, kernel_size=k[0], padding=k[0]//2)
        self.conv16 = conv_layer(chan*4, chan*4, kernel_size=k[0], padding=k[0]//2)
        
        self.conv21 = conv_layer(chn, chan, kernel_size=k[1], padding=k[1]//2)
        self.conv22 = conv_layer(chan, chan, kernel_size=k[1], padding=k[1]//2)
        self.conv23 = conv_layer(chan, chan*2, kernel_size=k[1], padding=k[1]//2)
        self.conv24 = conv_layer(chan*2, chan*2, kernel_size=k[1], padding=k[1]//2)
        self.conv25 = conv_layer(chan*2, chan*4, kernel_size=k[1], padding=k[1]//2)
        self.conv26 = conv_layer(chan*4, chan*4, kernel_size=k[1], padding=k[1]//2)
        
        self.conv31 = conv_layer(chn, chan, kernel_size=k[2], padding=k[2]//2)
        self.conv32 = conv_layer(chan, chan, kernel_size=k[2], padding=k[2]//2)
        self.conv33 = conv_layer(chan, chan*2, kernel_size=k[2], padding=k[2]//2)
        self.conv34 = conv_layer(chan*2, chan*2, kernel_size=k[2], padding=k[2]//2)
        self.conv35 = conv_layer(chan*2, chan*4, kernel_size=k[2], padding=k[2]//2)
        self.conv36 = conv_layer(chan*4, chan*4, kernel_size=k[2], padding=k[2]//2)
        
    def forward(self, x):
        x = [x]*3
        
        x[0] = self.conv11(x[0])
        x[0] = self.conv12(x[0])
        x[0] = self.pool(x[0])
        x[0] = self.dropout(x[0])
        x[0] = self.conv13(x[0])
        x[0] = self.conv14(x[0])
        x[0] = self.pool(x[0])
        x[0] = self.dropout(x[0])
        x[0] = self.conv15(x[0])
        x[0] = self.conv16(x[0])
        
        x[1] = self.conv21(x[1])
        x[1] = self.conv22(x[1])
        x[1] = self.pool(x[1])
        x[1] = self.dropout(x[1])
        x[1] = self.conv23(x[1])
        x[1] = self.conv24(x[1])
        x[1] = self.pool(x[1])
        x[1] = self.dropout(x[1])
        x[1] = self.conv25(x[1])
        x[1] = self.conv26(x[1])
        
        x[2] = self.conv31(x[2])
        x[2] = self.conv32(x[2])
        x[2] = self.pool(x[2])
        x[2] = self.dropout(x[2])
        x[2] = self.conv33(x[2])
        x[2] = self.conv34(x[2])
        x[2] = self.pool(x[2])
        x[2] = self.dropout(x[2])
        x[2] = self.conv35(x[2])
        x[2] = self.conv36(x[2])
        
        x = torch.cat(x, 1)
        
        x = self.fc(x)
        return x
    
class Sharkception(nn.Module):
    def __init__(self, chn):
        super().__init__()
        chan = 32
        k = [1,3,5,7]
        
        self.fc = dense(3072)
        
        self.pool = nn.MaxPool1d(2,2)
        self.dropout = nn.Dropout(.5)
        
        self.convs1 = nn.ModuleList([conv_layer(chn, chan, kernel_size=k[0], padding=k[0]//2),
                                     conv_layer(chan*2, chan, kernel_size=k[0], padding=k[0]//2),
                                     conv_layer(chan*2, chan, kernel_size=k[0], padding=k[0]//2),
                                     conv_layer(chan*2, chan*2, kernel_size=k[0], padding=k[0]//2),
                                     conv_layer(chan*4, chan*2, kernel_size=k[0], padding=k[0]//2),
                                     conv_layer(chan*4, chan*2, kernel_size=k[0], padding=k[0]//2),
                                     conv_layer(chan*4, chan*4, kernel_size=k[0], padding=k[0]//2),
                                     conv_layer(chan*8, chan*4, kernel_size=k[0], padding=k[0]//2),
                                     conv_layer(chan*8, chan*4, kernel_size=k[0], padding=k[0]//2),
                                     conv_layer(chan*8, chan*8, kernel_size=k[0], padding=k[0]//2),
                                     conv_layer(chan*16, chan*8, kernel_size=k[0], padding=k[0]//2),
                                     conv_layer(chan*16, chan*8, kernel_size=k[0], padding=k[0]//2)
                                    ])
                       
        self.convs2 = nn.ModuleList([conv_layer(chn, chan, kernel_size=k[1], padding=k[1]//2),
                                     conv_layer(chan*2, chan, kernel_size=k[1], padding=k[1]//2),
                                     conv_layer(chan*2, chan, kernel_size=k[1], padding=k[1]//2),
                                     conv_layer(chan*2, chan*2, kernel_size=k[1], padding=k[1]//2),
                                     conv_layer(chan*4, chan*2, kernel_size=k[1], padding=k[1]//2),
                                     conv_layer(chan*4, chan*2, kernel_size=k[1], padding=k[1]//2),
                                     conv_layer(chan*4, chan*4, kernel_size=k[1], padding=k[1]//2),
                                     conv_layer(chan*8, chan*4, kernel_size=k[1], padding=k[1]//2),
                                     conv_layer(chan*8, chan*4, kernel_size=k[1], padding=k[1]//2),
                                     conv_layer(chan*8, chan*8, kernel_size=k[1], padding=k[1]//2),
                                     conv_layer(chan*16, chan*8, kernel_size=k[1], padding=k[1]//2),
                                     conv_layer(chan*16, chan*8, kernel_size=k[1], padding=k[1]//2)
                                    ])
                       
        self.convs3 = nn.ModuleList([conv_layer(chn, chan, kernel_size=k[2], padding=k[2]//2),
                                     conv_layer(chan*2, chan, kernel_size=k[2], padding=k[2]//2),
                                     conv_layer(chan*2, chan, kernel_size=k[2], padding=k[2]//2),
                                     conv_layer(chan*2, chan*2, kernel_size=k[2], padding=k[2]//2),
                                     conv_layer(chan*4, chan*2, kernel_size=k[2], padding=k[2]//2),
                                     conv_layer(chan*4, chan*2, kernel_size=k[2], padding=k[2]//2),
                                     conv_layer(chan*4, chan*4, kernel_size=k[2], padding=k[2]//2),
                                     conv_layer(chan*8, chan*4, kernel_size=k[2], padding=k[2]//2),
                                     conv_layer(chan*8, chan*4, kernel_size=k[2], padding=k[2]//2),
                                     conv_layer(chan*8, chan*8, kernel_size=k[2], padding=k[2]//2),
                                     conv_layer(chan*16, chan*8, kernel_size=k[2], padding=k[2]//2),
                                     conv_layer(chan*16, chan*8, kernel_size=k[2], padding=k[2]//2)
                                    ])
                       
        self.convs4 = nn.ModuleList([conv_layer(chn, chan, kernel_size=k[3], padding=k[3]//2),
                                     conv_layer(chan*2, chan, kernel_size=k[3], padding=k[3]//2),
                                     conv_layer(chan*2, chan, kernel_size=k[3], padding=k[3]//2),
                                     conv_layer(chan*2, chan*2, kernel_size=k[3], padding=k[3]//2),
                                     conv_layer(chan*4, chan*2, kernel_size=k[3], padding=k[3]//2),
                                     conv_layer(chan*4, chan*2, kernel_size=k[3], padding=k[3]//2),
                                     conv_layer(chan*4, chan*4, kernel_size=k[3], padding=k[3]//2),
                                     conv_layer(chan*8, chan*4, kernel_size=k[3], padding=k[3]//2),
                                     conv_layer(chan*8, chan*4, kernel_size=k[3], padding=k[3]//2),
                                     conv_layer(chan*8, chan*8, kernel_size=k[3], padding=k[3]//2),
                                     conv_layer(chan*16, chan*8, kernel_size=k[3], padding=k[3]//2),
                                     conv_layer(chan*16, chan*8, kernel_size=k[3], padding=k[3]//2)
                                    ])
        
        self.reduce_filters = nn.ModuleList([conv_layer(chan*4, chan*2, kernel_size=1),
                                             conv_layer(chan*4, chan*2, kernel_size=1),
                                             conv_layer(chan*4, chan*2, kernel_size=1),
                                             conv_layer(chan*8, chan*4, kernel_size=1),
                                             conv_layer(chan*8, chan*4, kernel_size=1),
                                             conv_layer(chan*8, chan*4, kernel_size=1),
                                             conv_layer(chan*16, chan*8, kernel_size=1),
                                             conv_layer(chan*16, chan*8, kernel_size=1),
                                             conv_layer(chan*16, chan*8, kernel_size=1),
                                             conv_layer(chan*32, chan*16, kernel_size=1),
                                             conv_layer(chan*32, chan*16, kernel_size=1),
                                             conv_layer(chan*32, chan*16, kernel_size=1)])
                       
        
    def inception_module(self, x, i):
        x = [x]*4
        
        x[0] = self.convs1[i](x[0])
        x[1] = self.convs2[i](x[1])
        x[2] = self.convs3[i](x[2])
        x[3] = self.convs4[i](x[3])
       
        x = torch.cat(x, 1)
        
        return self.reduce_filters[i](x)
        
    def forward(self, x):
        for i in range(3):
            x = self.inception_module(x, i)
            
        x = self.pool(x)
        x = self.dropout(x)
        
        for i in range(3,6):
            x = self.inception_module(x, i)
            
        x = self.pool(x)
        x = self.dropout(x)
        
        for i in range(6,9):
            x = self.inception_module(x, i)
            
        x = self.pool(x)
        x = self.dropout(x)
        
        for i in range(9,12):
            x = self.inception_module(x, i)
        
        x = self.fc(x)
        
        return x
        
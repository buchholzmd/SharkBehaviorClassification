import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Dense(nn.Module):
    def __init__(self, dims):
        super(Dense, self).__init__()

        dims = [x for x in zip(dims, dims[1:])]

        self.dense = nn.Sequential(
            Flatten(),
            *list(nn.Linear(in_dim, out_dim) for in_dim, out_dim in dims)
        )
        
    def forward(self, x):
        return self.dense(x)
    
class ConvLayer(nn.Module):
    def __init__(self, 
                 in_chan, 
                 out_chan, 
                 kernel_size, 
                 stride, 
                 pad, 
                 transposed=False,
                 temporal=False):
        
        super(ConvLayer, self).__init__()

        if temporal:
            bn = nn.BatchNorm1d

            if not transposed:
                conv = nn.Conv1d
            else:
                conv = nn.ConvTranspose1d
        else:
            bn = nn.BatchNorm2d

            if not transposed:
                conv = nn.Conv2d
            else:
                conv = nn.ConvTranspose2d

        if not transposed:
            self.conv = conv(in_channels=in_chan, 
                             out_channels=out_chan, 
                             kernel_size=kernel_size, 
                             stride=stride, 
                             padding=pad,
                             bias=False
                             )
        else:
            self.conv = conv(in_channels=in_chan, 
                             out_channels=out_chan, 
                             kernel_size=kernel_size, 
                             stride=stride,
                             padding=pad,
                             output_padding=1,
                             bias=False
                             )

        self.prelu = nn.PReLU(out_chan)
        self.bn = bn(out_chan)
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if (isinstance(m, nn.Conv1d) or 
                isinstance(m, nn.ConvTranspose1d) or
                isinstance(m, nn.Conv2d) or 
                isinstance(m, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        x = self.bn(x)
        
        return x
    
class RecurrentConvolutionalLayer(nn.Module):
    def __init__(self, chanIn, chanOut, ks, pad, time_steps):
        super(RecurrentConvolutionalLayer, self).__init__()
        
        self.pad = False
        
        self.asym_pad_left = False
        self.asym_pad_top  = False
        self.asym_pad      = False
        
        if type(ks) is tuple:
            self.pad = True
            if not ks[0] % 2:
                self.asym_pad_left = True
                self.padl = nn.ZeroPad2d((pad[0],0,0,0))
                pad = 0, pad[1]
                
            if not ks[1] % 2:
                self.asym_pad_top = True
                self.padt = nn.ZeroPad2d((0,pad[1],0,0))
                pad =  pad[0], 0
        elif ks % 2:
            self.pad = True
            self.asym_pad = True
            if type(pad) is tuple:
                self.padlt = nn.ZeroPad2d((pad[0],0,pad[1],0))
                pad = (0,0)
            else:
                self.padlt = nn.ZeroPad2d((pad,0,pad,0))
                pad = 0
        
        self.conv1 = ConvLayer(chanIn, chanOut, kernel_size=ks, stride=(1,1), pad=pad)
        self.rcl = ConvLayer(chanOut, chanOut, kernel_size=ks, stride=(1,1), pad=pad)
        
        self.lrn = nn.LocalResponseNorm(2)
        
        self.time_steps = time_steps
        
    def forward(self, x):
        if self.pad:
            x = self.asymetric_padding(x)
        x = self.conv1(x)
        
        x_rec = x.clone()
        
        if self.pad:
            x = self.asymetric_padding(x)
        x = self.rcl(x)
        
        x += x_rec
        x = self.lrn(x)
        
        for i in range(1, self.time_steps):
            if self.pad:
                x = self.asymetric_padding(x)
            x = self.rcl(x)

            x += x_rec
            x = self.lrn(x)
        
        return x
    
    def asymetric_padding(self, x):
        if self.asym_pad_left:
            x = self.padl(x)
        elif self.asym_pad_top:
            x = self.padt(x)
        elif self.asym_pad:
            x = self.padlt(x)
            
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_chan, 
                       out_chan, 
                       kernel_size, 
                       stride, 
                       pad,
                       dim_red=False,
                       atrous=False):
        super(ResidualBlock, self).__init__()
        self.dim_red = dim_red
        
        if dim_red:
            stride1 = stride+1
        else: stride1 = stride
        if atrous:
            self.conv1 = nn.Conv2d(in_channels=in_chan, 
                                   out_channels=out_chan, 
                                   kernel_size=kernel_size, 
                                   stride=stride1, 
                                   padding=pad,
                                   dilation=2,
                                   bias=False
                                   )
            
            self.conv2 = nn.Conv2d(in_channels=out_chan, 
                                   out_channels=out_chan, 
                                   kernel_size=kernel_size, 
                                   stride=stride, 
                                   padding=pad,
                                   dilation=2,
                                   bias=False
                                   )
        else:
            self.conv1 = nn.Conv2d(in_channels=in_chan, 
                                   out_channels=out_chan, 
                                   kernel_size=kernel_size, 
                                   stride=stride1,
                                   padding=pad,
                                   bias=False
                                   )
            
            self.conv2 = nn.Conv2d(in_channels=out_chan, 
                                   out_channels=out_chan, 
                                   kernel_size=kernel_size, 
                                   stride=stride,
                                   padding=pad,
                                   bias=False
                                   )

        self.prelu1 = nn.PReLU(out_chan)
        self.prelu2 = nn.PReLU(out_chan)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        
        if dim_red:
            self.conv1x1 = nn.Conv2d(in_channels=in_chan, 
                                     out_channels=out_chan, 
                                     kernel_size=1, 
                                     stride=2,
                                     padding=0,
                                     bias=False
                                     )

            self.prelu3 = nn.PReLU(out_chan)
            self.bn3 = nn.BatchNorm2d(out_chan)
            
        self.init_weights()
            
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        nn.init.constant_(self.bn2.weight, 0) # zero initialize final bn
                
    def forward(self, x):
        if self.dim_red:
            x_res = self.conv1x1(x)
            x_res = self.prelu3(x_res)
            x_res = self.bn3(x_res)
        else:   
            x_res = x.clone()
    
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += x_res
        
        x = self.prelu2(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, proj_dim):
        super(Attention, self).__init__()
        self.W_k = nn.Linear(2*hidden_size, proj_dim, bias=False)
        self.W_q = nn.Linear(2*hidden_size, proj_dim, bias=False)
        self.V   = nn.Linear(proj_dim, 1, bias=False)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, query, key, value):
        query = self.dropout(self.W_q(query))
        key = self.dropout(self.W_k(key))

        query = query.unsqueeze(2)
        key = key.unsqueeze(1)
        
        features = torch.tanh(query + key)
        
        scores = self.V(features)
        scores = scores.squeeze()

        attn_weights = F.softmax(scores, 1)
        attn_weights = attn_weights.unsqueeze(1)
        
        return torch.bmm(attn_weights, value), attn_weights

class NonLocalBlock(nn.Module):
    def __init__(self, in_chan, dim=2):
        super(NonLocalBlock, self).__init__()

        # self.proj_chan = in_chan // 2
        self.proj_chan = in_chan

        if dim == 1:
            conv = nn.Conv1d
            pool = nn.MaxPool1d(kernel_size=2)
            bn = nn.BatchNorm1d
        elif dim == 2:
            conv = nn.Conv2d
            pool = nn.MaxPool2d(kernel_size=(2,2))
            bn = nn.BatchNorm2d

        self.theta = conv(in_channels=in_chan, 
                          out_channels=self.proj_chan,
                          kernel_size=1, 
                          stride=1,
                          padding=0,
                          bias=False
                          )

        self.phi = conv(in_channels=in_chan, 
                        out_channels=self.proj_chan,
                        kernel_size=1, 
                        stride=1,
                        padding=0,
                        bias=False
                        )

        self.g = conv(in_channels=in_chan, 
                      out_channels=self.proj_chan,
                      kernel_size=1, 
                      stride=1,
                      padding=0,
                      bias=False
                      )

        self.W = conv(in_channels=self.proj_chan, 
                      out_channels=in_chan,
                      kernel_size=1, 
                      stride=1,
                      padding=0,
                      bias=False
                      )

        self.bn = bn(in_chan)
        self.pool = pool

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x)
        # g_x = self.pool(g_x)
        g_x = g_x.view(batch_size, self.proj_chan, -1)

        theta_x = self.theta(x)
        theta_x = theta_x.view(batch_size, self.proj_chan, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x)
        # phi_x = self.pool(phi_x)
        phi_x = phi_x.view(batch_size, self.proj_chan, -1)

        f_x = torch.bmm(theta_x, phi_x)
        f_x = F.softmax(f_x, dim=-1)
        f_x = f_x.permute(0, 2, 1)

        out = torch.bmm(g_x, f_x)
        out = out.view(batch_size, self.proj_chan, *x.shape[2:])
        # out = self.W(out)
        # out = self.bn(out)
        # out += x

        return out

     
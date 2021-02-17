import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.layers import (
    Attention,
    Dense
)

class SharkGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, attn_dim=64, fc_dim=512, attention=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.num_layers = num_layers
        self.fc_dim = fc_dim

        self.attention = attention # LOL
        
        self.gru = nn.GRU(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    dropout=.5,
                    bidirectional=True)

        if attention:
            self.attn = Attention(self.input_size, self.hidden_size, attn_dim) # 64
        
        self.fc1 = nn.Linear(hidden_size*2, self.fc_dim)
        self.bn1 = nn.BatchNorm1d(self.fc_dim)
        self.prelu1 = nn.PReLU(self.fc_dim)
        self.dp1 = nn.Dropout(.5)
        
        self.fc2 = nn.Linear(self.fc_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.prelu2 = nn.PReLU(64)
        
        self.fc3 = nn.Linear(64, self.output_size)
        self.bn3 = nn.BatchNorm1d(self.output_size)
        self.prelu3 = nn.PReLU(self.output_size)
        
        self.init_weights()
        
    def forward(self, x):
        output, hidden = self.gru(x, None)

        for h in hidden:
            if h.requires_grad:
                h.register_hook(lambda x: x.clamp(min=-10, max=10) if x is not None else x)
        
        if self.attention:
            batch_size = hidden.shape[1]
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            
            hidden = hidden[-1,...] # choose last layer of LSTM
            hidden = hidden.permute((1,0,2)) # batch first
            hidden = torch.cat([hidden[:,0,:], hidden[:,1,:]], dim=1) # concat forward & backward
            hidden = hidden.unsqueeze(1) # expand the dimension for addition
            
            context_vector, attn_weights = self.attn(hidden, output, output)

            x = context_vector.squeeze(1)

        else:
            x = output[:,-1, :]

        x = self.fc1(x)
        x = self.prelu1(x)
        x = self.bn1(x)
        
        x = self.dp1(x)
        
        x = self.fc2(x)
        x = self.prelu2(x)
        x = self.bn2(x)
        
        x = self.fc3(x)
        x = self.prelu3(x)
        x = self.bn3(x)
        
        return x

    def init_weights(self):
        for m in self.modules():
            if type(m) is nn.GRU:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for ih in param.chunk(3, 0):
                            for ih_dir in ih.chunk(2, 1):
                                torch.nn.init.orthogonal_(ih_dir)
                    elif 'weight_hh' in name:
                        for hh in param.chunk(3, 0):
                            torch.nn.init.orthogonal_(hh)
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param)
                       
                    # TODO... 
                    # init hidden bias as -1
                    # implement hidden dropout
                    # plot grad norms
                    
class SharkLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, fc_dim=512, attention=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.num_layers = num_layers
        self.fc_dim = fc_dim

        self.attention = attention
        
        self.lstm = nn.LSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    dropout=.5,
                    bidirectional=True)

        if attention:
            self.attn = Attention(self.input_size, self.hidden_size, 64)
        
        self.fc1 = nn.Linear(hidden_size*2, self.fc_dim)
        self.bn1 = nn.BatchNorm1d(self.fc_dim)
        self.prelu1 = nn.PReLU(self.fc_dim)
        self.dp1 = nn.Dropout(.5)
        
        self.fc2 = nn.Linear(self.fc_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.prelu2 = nn.PReLU(64)
        
        self.fc3 = nn.Linear(64, self.output_size)
        self.bn3 = nn.BatchNorm1d(self.output_size)
        self.prelu3 = nn.PReLU(self.output_size)
        
        self.init_weights()
        
    def forward(self, x):        
        output, hidden = self.lstm(x, None)
        
        for h in hidden:
            if h.requires_grad:
                h.register_hook(lambda x: x.clamp(min=-10, max=10) if x is not None else x)

        if self.attention:
            batch_size = hidden[0].shape[1]
            hidden = hidden[0].view(self.num_layers, 2, batch_size, self.hidden_size)
            
            hidden = hidden[-1,...] # choose last layer of LSTM
            hidden = hidden.permute((1,0,2)) # batch first
            hidden = torch.cat([hidden[:,0,:], hidden[:,1,:]], dim=1) # concat forward & backward
            hidden = hidden.unsqueeze(1) # expand the dimension for addition
            
            context_vector, attn_weights = self.attn(hidden, output, output)

            x = context_vector.squeeze(1)

        else:
            x = output[:,-1, :]

        x = self.fc1(x)
        x = self.prelu1(x)
        x = self.bn1(x)
        
        x = self.dp1(x)
        
        x = self.fc2(x)
        x = self.prelu2(x)
        x = self.bn2(x)
        
        x = self.fc3(x)
        x = self.prelu3(x)
        x = self.bn3(x)
        
        return x
    
    def init_weights(self):
        for m in self.modules():
            if type(m) is nn.LSTM:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for ih in param.chunk(4, 0):
                            torch.nn.init.orthogonal_(ih)
                    elif 'weight_hh' in name:
                        for hh in param.chunk(4, 0):
                            torch.nn.init.orthogonal_(hh)   
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param)

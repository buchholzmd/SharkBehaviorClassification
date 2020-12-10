import torch
import torch.nn as nn
import torch.nn.functional as F

class SharkGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, fc_dim=512):
        super().__init__()
        self.new_dim = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.num_layers = num_layers
        
        self.fc_dim = fc_dim
        
        self.gru = nn.GRU(
                    input_size=self.new_dim,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    dropout=.5,
                    bidirectional=True)
        
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
        r_out, hidden = self.gru(x, None)
        
        for h in hidden:
            if h.requires_grad:
                h.register_hook(lambda x: x.clamp(min=-10, max=10) if x is not None else x)
        
        x = self.fc1(r_out[:,-1, :])
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
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, fc_dim=512):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.num_layers = num_layers
        
        self.fc_dim = fc_dim
        
        self.lstm = nn.LSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    dropout=.5,
                    bidirectional=True)
        
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
        r_out, hidden = self.lstm(x, None)
        
        for h in hidden:
            if h.requires_grad:
                h.register_hook(lambda x: x.clamp(min=-10, max=10) if x is not None else x)
        
        x = self.fc1(r_out[:,-1, :])
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
    
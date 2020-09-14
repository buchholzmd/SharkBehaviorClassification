import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=True,
                            dropout=.5)
        
    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(2, x.shape[1], self.hidden_size),
                torch.zeros(2, x.shape[1], self.hidden_size))
        
class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers=2):
        super(AttentionDecoder, self).__init__()

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.fc = nn.Linear(self.hidden_size*3, 1)
        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=False,
                            dropout=.5)

    def forward(self, x, hidden):
        attn_weights = []
        for i in range(len(x)):
            attn_weights.append(self.fc(torch.cat((hidden[0][0], x[i]),
                                                  dim=1)))
            
        attn_weights = F.softmax(torch.cat(attn_weights,1),1)
        
        print(attn_weights.shape)
        print(attn_weights.unsqueeze(1).shape)
        print(x.shape)
        print(x.view(1,-1,self.hidden_size).shape)
        x = torch.bmm(attn_weights.unsqueeze(1),
                      x.view(1,-1,self.hidden_size))
                                          
        output, hidden = self.lstm(x.unsqueeze(0), hidden)
                                          
        return output, hidden, attn_weights

class SharkAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, fc_dim=512):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.num_layers = num_layers
        
        self.fc_dim = fc_dim
                                          
        self.enc = Encoder(self.input_size, 
                           self.hidden_size, 
                           num_layers=self.num_layers)
                                          
        self.dec = AttentionDecoder(self.hidden_size, 
                                    num_layers=self.num_layers)
        
        self.fc1 = nn.Linear(hidden_size*2, self.fc_dim)
        self.bn1 = nn.BatchNorm1d(self.fc_dim)
        self.dp1 = nn.Dropout(.5)
        self.fc2 = nn.Linear(self.fc_dim, self.output_size)
        self.bn2 = nn.BatchNorm1d(self.output_size)
        
    def forward(self, x):        
        x, hidden = self.enc(x, None)
        x, hidden, attn_weights = self.dec(x, hidden)            
        
        x = self.fc1(x[:,-1, :])
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.dp1(x)
        
        x = self.fc2(x)        
        x = self.bn2(x)
        x = F.relu(x)    
        return x

'''
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
                    bidirectional = True,
                    dropout=0.5)
        
        self.fc1 = nn.Linear(hidden_size*2, self.fc_dim)
        self.bn1 = nn.BatchNorm1d(self.fc_dim)
        self.dp1 = nn.Dropout(.5)
        self.fc2 = nn.Linear(self.fc_dim, self.output_size)
        self.bn2 = nn.BatchNorm1d(self.output_size)
        self.init_weights()
        
    def forward(self, x):
        r_out, hidden = self.gru(x, None)
        
        x = self.fc1(r_out[:,-1, :])
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.dp1(x)
        x = self.fc2(x)        
        x = self.bn2(x)
        x = F.relu(x)
        
        return x

    def init_weights(self):
        for m in self.modules():
            if type(m) is nn.GRU:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for ih in param.chunk(3, 0):
                            for ih_dir in ih.chunk(2, 1):
                                torch.nn.init.xavier_uniform_(ih_dir)
                    elif 'weight_hh' in name:
                        for hh in param.chunk(3, 0):
                            torch.nn.init.orthogonal_(hh)
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param)
                       
                    # TODO... 
                    # init hidden bias as -1
                    # implement hidden dropout
                    # plot grad norms             
'''
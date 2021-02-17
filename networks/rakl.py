import torch
import gpytorch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from networks.layers import (
    Attention,
    Dense
)

class RecurrentKernelLearningModel(gpytorch.Module):
    def __init__(self, 
                 input_size,
                 hidden_size,
                 time_steps,
                 num_layers,
                 attention,
                 attn_dim=None,
                 grid_bounds=(-10., 10.)):
        super(RecurrentKernelLearningModel, self).__init__()

        self.attention = attention

        self.grid_bounds = grid_bounds
        self.num_dim = 2*hidden_size

        self.seq_len = time_steps

        if attention:
            self.feature_extractor = GRUFeatureExtractor(input_size, 
                                                         hidden_size,
                                                         num_layers=num_layers,
                                                         attention=attention,
                                                         attn_dim=attn_dim)
        else:
            self.feature_extractor = GRUFeatureExtractor(input_size, 
                                                         hidden_size,
                                                         num_layers=num_layers,
                                                         attention=attention)

        self.gp_layer = GaussianProcessLayer(num_dim=self.num_dim, grid_bounds=self.grid_bounds)

    def forward(self, x):
        if self.attention:
            features, attn_weights = self.feature_extractor(x)

            batch_size = features.size(0)
            features = features.reshape((self.seq_len * batch_size, -1))
        else:
            features = self.feature_extractor(x)

        features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)

        res = self.gp_layer(features)

        if self.attention:
            return res, attn_weights
        else:
            return res

class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    np.exp(-1), np.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean, covar)

class GRUFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4, attention=False, attn_dim=64):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.num_layers = num_layers

        self.attention = attention # LOL
        
        self.gru = nn.GRU(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    dropout=.5,
                    bidirectional=True)

        if attention:
            assert(attn_dim is not None)
            self.attn = Attention(self.input_size, self.hidden_size, attn_dim)
        
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

            x = torch.cat([output[:,:-1,:], context_vector], dim=1)

            return x, attn_weights

        return output[:,-1,:]

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
                       
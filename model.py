"""
Models initially adapted from https://github.com/jeremyhengjm/CompDoobTransform
"""

import torch

class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final = False, activation_fn = torch.nn.LeakyReLU):
        """
        Parameters
        ----------    
        input_dim : int specifying input_dim of input 

        layer_widths : list specifying width of each layer 
            (len is the number of layers, and last element is the output input_dim)

        activate_final : bool specifying if activation function is applied in the final layer

        activation_fn : activation function for each layer        
        """
        super().__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            # # same init for everyone
            # torch.nn.init.constant_(layers[-1].weight, 0)
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x

class V0_Network_multi(torch.nn.Module):

    def __init__(self, dimension_state, dimension_obs, number_obs, config):
        """
        Parameters
        ----------
        dimension_state : int specifying state dimension
        dimension_obs : int specifying observation dimension
        config : dict containing      
            layers : list specifying width of each layer 
        """
        super().__init__()
        input_dimension = dimension_state + dimension_obs*number_obs
        self.number_obs = number_obs        
        layers = config['layers']
        self.standardization = config['standardization']
        self.net = MLP(input_dimension, 
                       layer_widths = layers + [1],
                       activate_final = False,
                       activation_fn = torch.nn.LeakyReLU())

    def forward(self, x, y):
        """
        Parameters
        ----------
        x : state (B, d)
        
        y : observations (B, number_obs, dimension_obs)
                        
        Returns
        -------    
        out :  output (B)
        """
        
        B = x.shape[0]
        Nobs = y.shape[1]
        d_obs = y.shape[2]
        y_ = y.reshape((B, Nobs*d_obs))

        x_c = (x - self.standardization['x_mean']) / self.standardization['x_std']
        y_c = (y_ - self.standardization['y_mean']) / self.standardization['y_std']            
        
        h = torch.cat([x_c, y_c], -1) # size (B, d+nobs*p)
        out = self.net(h).squeeze(-1) # size (B)

        return out

class Z_Network_multi(torch.nn.Module):

    def __init__(self, dimension_state, dimension_noise, dimension_obs, number_obs, config):
        """
        Parameters
        ----------
        dimension_state : int specifying state dimension
        dimension_obs : int specifying observation dimension
        config : dict containing
            layers : list specifying width of each layer
        """
        super().__init__()
        input_dimension = dimension_state + dimension_obs*number_obs + 1
        self.number_obs = number_obs
        layers = config['layers']
        self.standardization = config['standardization']
        net_layers = []
        prev_width = input_dimension
        for layer_width in layers:
            net_layers.append(torch.nn.Linear(prev_width, layer_width))
            net_layers.append(torch.nn.LeakyReLU())
            prev_width = layer_width
        net_layers.append(torch.nn.Linear(prev_width, dimension_noise))
        self.net = torch.nn.Sequential(*net_layers)

    def forward(self, t, x, y):
        """
        Parameters
        ----------
        t : shared scalar time, repeated across the batch

        x : state (B, d)

        y : observations (B, number_obs, dimension_obs)
                        
        Returns
        -------    
        out :  output (B, dimension_noise)
        """
        
        B = x.shape[0]
        t_ = t.repeat((B, 1))
        Nobs = y.shape[1]
        d_obs = y.shape[2]
        y_ = y.reshape((B, Nobs*d_obs))

        x_c = (x - self.standardization['x_mean']) / self.standardization['x_std']
        y_c = (y_ - self.standardization['y_mean']) / self.standardization['y_std']            
        
        h = torch.cat([t_, x_c, y_c], -1) # size (B, 1+d+nobs*p)
        out = self.net(h) # size (B, dimension_noise)
        
        return out

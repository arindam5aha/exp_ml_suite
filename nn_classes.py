"""Neural-network building blocks and training helpers.

This module includes:
- a configurable multi-layer perceptron (`MLP`),
- a Gaussian latent head variant (`GaussianMLP`),
- activation and weight-initialization utilities,
- distribution helpers,
- a lightweight training loop utility.

Written by: Arindam Saha, ANU, 2025.
GitHub: arindam5aha, arindam.saha@anu.edu.au
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

class MLP(nn.Module):
    """Configurable feed-forward multilayer perceptron.

    Args:
        input_size: Input feature dimension.
        output_sizes: Output specification. Can be:
            - `int` for a single head,
            - `list[int]` for multiple output heads,
            - `dict[str, int]` for named output heads.
        hidden_sizes: Hidden-layer widths.
        hidden_activation: Hidden activation name or per-layer list.
        output_activation: Output activation name or per-head list.
        use_dropout: Whether to apply dropout after hidden activations.
        dropout: Dropout module.
        hidden_init: Hidden-layer weight initialization method.
        bias_init: Hidden-layer bias initialization value.
        init_args: Extra args used by selected hidden initializer.
        output_init: Output-layer weight initialization method.
        output_bias_init: Output-layer bias initialization value.
        output_init_args: Extra args used by selected output initializer.
        layer_norm: Reserved flag (currently not implemented).
        return_last_hidden: If `True`, also return final hidden features.
        concat_inputs: If `True`, concatenates tuple inputs on last dim.
    """

    def __init__(self, 
                 input_size:int, 
                 output_sizes:int|list|dict, 
                 hidden_sizes:int|list=[256, 256, 256], 
                 hidden_activation:str|list='relu',
                 output_activation:str|list='identity',
                 use_dropout:bool=False,
                 dropout=nn.Dropout(0.2),
                 hidden_init:str='kaiming',
                 bias_init=0.0,
                 init_args=None,
                 output_init:str='uniform',
                 output_bias_init=0.0,
                 output_init_args=(-3e-6, 3e-6),
                 layer_norm=False,
                 return_last_hidden=False,
                 concat_inputs=False):
        super().__init__()

        if isinstance(output_sizes, int):
            output_sizes = [output_sizes]
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.layers = nn.ModuleList()
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            x = nn.Linear(in_size, next_size)
            layer_initialiser(x, hidden_init, bias_init, init_args)
            self.layers.append(x)
            in_size = next_size
            if layer_norm:
                raise NotImplementedError
            
        self.out_layer = nn.ModuleList()
        if isinstance(output_sizes, list):
            self.out_keys = None
            for size in output_sizes:
                x = nn.Linear(in_size, size)
                layer_initialiser(x, output_init, output_bias_init, output_init_args)
                self.out_layer.append(x)
        if isinstance(output_sizes, dict):
            self.out_keys = []
            for key, size in output_sizes.items():
                x = nn.Linear(in_size, size)
                layer_initialiser(x, output_init, output_bias_init, output_init_args)
                self.out_layer.append(x)
                self.out_keys.append(key)

        self.hidden_activation = activation_from_string(hidden_activation, len(hidden_sizes))
        self.output_activation = activation_from_string(output_activation, len(output_sizes))
        self.use_dropout = use_dropout
        self.dropout = dropout
        self.return_last_hidden = return_last_hidden
        self.concat_inputs = concat_inputs

    def forward(self, input):
        """Run a forward pass.

        Args:
            input: Tensor input, or tuple of tensors when `concat_inputs=True`.

        Returns:
            One of the following based on output configuration:
            - single tensor,
            - tuple of tensors,
            - dict of tensors.

            If `return_last_hidden=True`, returns `(output, last_hidden)`.
        """
        h = input
        if self.concat_inputs:
            h = torch.cat(h, -1)
        for i, x in enumerate(self.layers):
            h = x(h)
            h = self.hidden_activation[i](h)
            if self.use_dropout and self.dropout is not None:
                h = self.dropout(h)
        
        if len(self.out_layer) > 1:
            output = tuple(self.output_activation[i](x(h)) for i, x in enumerate(self.out_layer))
            if self.out_keys is not None:
                output = dict(zip(self.out_keys, output))
        else:
            output = self.output_activation[0](self.out_layer[0](h))
        if self.return_last_hidden:
            return output, h
        else:
            return output     
        

class GaussianMLP(MLP):
    """MLP that parameterizes a Gaussian latent distribution.

    The network produces mean and standard-deviation heads and returns
    a reparameterized sample along with distribution parameters.

    Returned dictionary keys:
        - `sample`
        - `mean`
        - `std`
    """

    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 hidden_sizes=[256, 256, 256],
                 hidden_activation='relu',
                 latent_activation='identity',
                 use_dropout=False,
                 dropout=nn.Dropout(0.2),
                 batch_norm_mean:bool=False,
                 hidden_init='kaiming',
                 bias_init=0.0,
                 init_args=None,
                 output_init='uniform',
                 output_bias_init=0.0,
                 output_init_args=(-3e-6, 3e-6),
                 layer_norm=False,
                 clamp_latent=False,
                 mu_range=[-10, 10],
                 std_range=[1e-2, 10],
                 return_last_hidden=False,
                 independent_normal=True):
        super().__init__(input_size,
                         [output_size, output_size],
                         hidden_sizes,
                         hidden_activation,
                         latent_activation,
                         use_dropout,
                         dropout,
                         hidden_init,
                         bias_init,
                         init_args,
                         output_init,
                         output_bias_init,
                         output_init_args,
                         layer_norm,
                         return_last_hidden)
        
        if batch_norm_mean:
            self.BN1d = nn.BatchNorm1d(output_size)
        
        self.batch_norm_mean = batch_norm_mean
        self.clamp = clamp_latent
        self.mu_range = mu_range
        self.std_range = std_range
        self.return_last_hidden = return_last_hidden
        self.independent_normal = independent_normal
        
    def forward(self, input) -> dict:
        """Run a forward pass and sample from the latent Gaussian.

        Args:
            input: Input tensor.

        Returns:
            Dict with keys `sample`, `mean`, and `std`.
            If `return_last_hidden=True`, returns `(latent_dict, last_hidden)`.
        """
        if self.return_last_hidden:
            (mu, std), last_h = super().forward(input)
        else:
            mu, std = super().forward(input)
        
        std = nn.Softplus()(std) + self.std_range[0]

        if self.clamp:
            if self.mu_range is not None:
                mu = torch.clamp(mu, self.mu_range[0], self.mu_range[1])
            if self.std_range is not None:
                std = torch.clamp(std, self.std_range[0], self.std_range[1])
        
        if self.batch_norm_mean:
            mu = self.BN1d(mu)
        
        if self.independent_normal:
            sample = normal_dist(mu, std).rsample()
        else:
            sample = reparameterize(mu, std)
        latent = {'sample': sample, 'mean': mu, 'std': std}
        
        if self.return_last_hidden:
            return latent, last_h
        else:
            return latent
        
str_to_activation = {
    'identity': lambda x: x,
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leakyrelu': nn.LeakyReLU(0.2),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'swish': nn.SiLU()
}
        
        
def activation_from_string(input:str|list, num_layers:int=None) -> list:
    """Map activation name(s) to activation module/function list.

    Args:
        input: Activation name or list of names.
        num_layers: Expected output list length.

    Returns:
        List of callables/activation modules with length `num_layers`.
    """
    if isinstance(input, str):
        if num_layers is None:
            num_layers = 1
        return [str_to_activation[input]]*num_layers
    elif isinstance(input, list):
        if num_layers is None:
            num_layers = len(input)
        assert len(input) == num_layers
        acts = []
        for x in input:
            acts.append(str_to_activation[x])
        return acts
    
# NN weights and biases initialiser ###########################

def layer_initialiser(layer, method, b_val, args):
    """Initialize a linear layer's weights and bias.

    Args:
        layer: Layer with `weight` and `bias` parameters.
        method: One of `kaiming`, `uniform`, `xavier`, `orthogonal`.
        b_val: Constant bias value.
        args: Additional initializer arguments:
            - `uniform`: tuple `(a, b)`
            - `xavier`: gain value
            - `orthogonal`: gain value
            - `kaiming`: ignored
    """

    if method == 'kaiming':
        nn.init.kaiming_uniform_(layer.weight)

    elif method == 'uniform':
        a, b = args
        nn.init.uniform_(layer.weight, a, b)        

    elif method == 'xavier':
        nn.init.xavier_normal_(layer.weight, gain=args)

    elif method == 'orthogonal':
        nn.init.orthogonal_(layer.weight, gain=args)
    
    else:
        raise NotImplementedError
    
    nn.init.constant_(layer.bias, b_val)
    
    
def normal_dist(mean, std, mean_scale=1, init_std=0.0, min_std=0.01, mean_activation:str=None, event_shape=1):
    """Create an Independent Normal distribution from mean/std tensors.

    Args:
        mean: Mean tensor.
        std: Standard-deviation tensor.
        mean_scale: Scale divisor applied to mean.
        init_std: Reserved argument for legacy compatibility.
        min_std: Reserved argument for legacy compatibility.
        mean_activation: Optional activation name applied to mean.
        event_shape: Number of rightmost dims treated as event dims.

    Returns:
        `torch.distributions.Independent(torch.distributions.Normal(...))`.
    """
    mean = mean/mean_scale
    if mean_activation is not None:
        activation = str_to_activation[mean_activation]
        mean = activation(mean)
    #std = F.softplus(std + init_std) + min_std
    dist = td.Normal(mean, std)
    return td.Independent(dist, event_shape)

def get_dist(state):
    """Build a distribution from latent state dict.

    Args:
        state: Dict containing at least `mean` and `std` tensors.

    Returns:
        Independent normal distribution.
    """
    return normal_dist(state['mean'], state['std'])
    
def reparameterize(mean, std, logvar=False):
    """Sample using the reparameterization trick.

    Args:
        mean: Mean tensor.
        std: Std tensor, or log-variance tensor when `logvar=True`.
        logvar: If `True`, interprets `std` argument as log-variance.

    Returns:
        Reparameterized sample tensor.
    """
    if logvar:
        std = torch.exp(std/2)
    eps = torch.autograd.Variable(torch.randn_like(std))
    return mean + std*eps



def train(model, compute_loss, train_data, optimizer, validation_data=None, 
          beta=None, scheduler=None, epochs=500, batch_size=50, clip_grad_norm=None, plot=True):
    """Train a model with optional validation and loss plotting.

    Args:
        model: PyTorch model to optimize.
        compute_loss: Callable receiving a sliced mini-batch (and optional
            `beta`) and returning a scalar loss tensor.
        train_data: Sliceable training dataset.
        optimizer: PyTorch optimizer.
        validation_data: Optional sliceable validation dataset.
        beta: Optional extra argument passed into `compute_loss`.
        scheduler: Optional scheduler stepped once per epoch.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        clip_grad_norm: Optional max norm for gradient clipping.
        plot: If `True`, plots train/validation losses when validation data is
            provided.

    Notes:
        This helper assumes `train_data[j:j+batch_size]` style slicing is
        supported by both training and validation datasets.
    """
    
    disable = True
    if validation_data is not None:
        losses = np.full((epochs, 2), np.nan)
        disable = False
        
    with trange(epochs, desc="Training Model", disable = disable) as pbar:
        for epoch in pbar:
            running_loss = 0.0
            
            for j in range(0, len(train_data), batch_size):
                model.train()
                if beta is not None:
                    loss = compute_loss(train_data[j:j + batch_size], beta)
                else:
                    loss = compute_loss(train_data[j:j + batch_size])

                optimizer.zero_grad()
                loss.backward()
                
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                optimizer.step()
                running_loss += loss.item()
                
            running_loss = running_loss/(len(train_data)/ batch_size)
            
            if scheduler is not None:
                scheduler.step()

            if validation_data is not None:
                model.eval()
                validation_loss = 0.0
                for j in range(0, len(validation_data), batch_size):
                    with torch.no_grad():
                        if beta is not None:
                            test_loss = compute_loss(validation_data[j:j + batch_size], beta)
                        else:
                            test_loss = compute_loss(validation_data[j:j + batch_size])
                            
                    validation_loss += test_loss.item()
                    
                validation_loss = validation_loss/(len(validation_data)/ batch_size)
                losses[epoch] = [running_loss, validation_loss]
                
                pbar.set_postfix(train_loss=running_loss, validation_loss=validation_loss) # loss per batch
            
    # Plot losses only when validation data is also given
    if validation_data is not None and plot:
        plt.plot(losses[3:])
        plt.yscale("log")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend(["training", "validation"])
        plt.show()
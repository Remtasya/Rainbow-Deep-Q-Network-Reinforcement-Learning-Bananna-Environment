import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

class custom_linear_noisy_net(nn.Module):
    """
    This is a custom linear layer where the weight and bias term have been replaced with semi-learnable noise.
    
    The usual linear layer equation is 
    y = w*x+b 
    which instead becomes:
    y = (mu_w+sig_w*eps_w)*x+(mu_b+sig_b*eps_b)
    where mu and sig are learnable, while eps are unlearnable guassian noise,
    and _w have dimension input_size*output_size, while _b has dimension output_size.
    
    Note we also do a trick to reuse the guassian noise when generating eps:
    We first generate a N(0,1) RV eps for each input i, output j, so that we have eps(i) and eps(j), and then set:
    eps_w(i,j) = f(eps(i))*f(eps(j))
    eps_b(j) = f(eps(j))
    
    where f(x) = sign(x)*sqrt(abs(x))
    which makes the noise locally-independent wrt a given input-output.
    """
    def __init__(self, input_size, output_size,sigma_zero = 0.5):
        super(custom_linear_noisy_net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sigma_zero = sigma_zero
        # learnable parameters (mu and sigma for weight and bias)
        self.mu_w, self.sigma_w, self.mu_b, self.sigma_b = self.initialise_learnable_parameter()
        # non-learnable noise (epsilon for weight and bias)
        self.register_buffer('epsilon_i', torch.Tensor(input_size))
        self.register_buffer('epsilon_j', torch.Tensor(output_size))

    def initialise_learnable_parameter(self):
        # I've used the same initialisations used for the weights and bias of the pytorch standard linear layer
        # hopefully this will cause minimal unintended changes to learning
        bound = 1/math.sqrt(self.input_size)
        
        mu_w = Parameter(torch.Tensor(self.output_size, self.input_size))
        sigma_w = Parameter(torch.Tensor(self.output_size, self.input_size).fill_(self.sigma_zero*bound))
        mu_b = Parameter(torch.Tensor(self.output_size))
        sigma_b = Parameter(torch.Tensor(self.output_size).fill_(self.sigma_zero*bound))
                            
        nn.init.uniform_(mu_w, -bound, bound)
        nn.init.uniform_(mu_b, -bound, bound)
                            
        #learnable_parameter.requires_grad = True
        return mu_w, sigma_w, mu_b, sigma_b

    def refresh_nonlearnable_parameters(self):
        # every forward pass we must refresh the nonlearnable noise
        torch.randn(self.input_size, out=self.epsilon_i)
        torch.randn(self.output_size, out=self.epsilon_j)
        
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        epsilon_i, epsilon_j = func(self.epsilon_i), func(self.epsilon_j)
        
        epsilon_w = epsilon_j.unsqueeze(1)*epsilon_i.unsqueeze(0)
        epsilon_b = epsilon_j
        return epsilon_w, epsilon_b
        
    def forward(self,batch_data):
        # calculating the noisy weight and bias equation, i.e. y=w*x+b => y=(mu_w+sig_w*eps_w)*x+(mu_b+sig_b*eps_b)
        epsilon_w, epsilon_b = self.refresh_nonlearnable_parameters()
        noisy_weight = self.mu_w+self.sigma_w*epsilon_w
        noisy_bias = self.mu_b+self.sigma_b*epsilon_b
        output =  F.linear(batch_data, noisy_weight, noisy_bias)
        return output
    
    def extra_repr(self):
        return 'input_size={}, output_size={}'.format(
            self.input_size, self.output_size)
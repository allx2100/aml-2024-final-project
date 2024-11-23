import torch
from torch import nn
from torch.nn import functional as F
from models.base import BaseVAE

class DiagVAE(BaseVAE):
    def __init__(self, input_size, latent_dim, hidden_dims=[512, 256, 128, 64, 32], kld_weight=1):
        super(DiagVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.kld_weight = kld_weight

        self.input_layer = nn.Linear(input_size, hidden_dims[0])

        modules = []
        for i in range(len(self.hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]),
                    nn.LeakyReLU())
            )

        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.var = nn.Linear(hidden_dims[-1], latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        modules = []
        hidden_dims.reverse()
        for i in range(len(self.hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_dims[-1], input_size)

    def encode(self, input):
        result = self.input_layer(input)
        result = self.encoder(result)
        result = torch.flatten(result, start_dim=1)
  
        mu = self.mu(result)
        log_var = self.var(result)

        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), input, mu, log_var

    def loss_function(self, args):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 -log_var.exp(), dim=1))

        loss = recons_loss + self.kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':kld_loss.detach()}

    def reconstruct(self, input):
        mu, _ = self.encode(input)
        return self.decode(mu)
    
        
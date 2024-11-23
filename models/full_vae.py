import torch
from models.base import BaseVAE
from torch import nn
from torch.nn import functional as F

class FullVAE(BaseVAE):
    def __init__(self, input_size, latent_dim, hidden_dims=[128, 64, 32], kld_weight=1):
        super(FullVAE, self).__init__()

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
        self.m = nn.Linear(hidden_dims[-1], latent_dim)
        self.M = nn.Linear(hidden_dims[-1], int(latent_dim * (latent_dim - 1) / 2))

        self.cov_idx_row, self.cov_idx_col = torch.tril_indices(self.latent_dim, self.latent_dim)
        non_diag = (self.cov_idx_row != self.cov_idx_col)
        self.cov_idx_col = self.cov_idx_col[non_diag]
        self.cov_idx_row = self.cov_idx_row[non_diag]

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
        m = self.m(result)
        M = self.M(result)

        return mu, m, M

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def get_cov(self, m, M):
        L = torch.zeros(len(m), self.latent_dim, self.latent_dim)
        L[:, self.cov_idx_row, self.cov_idx_col] = M
        L += torch.diag_embed(torch.exp(m))

        return L

    def reparameterize(self, mu, L):
        eps = torch.randn((len(mu), self.latent_dim))
        return mu + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)


    def forward(self, input):
        mu, m, M = self.encode(input)
        L = self.get_cov(m, M)
        z = self.reparameterize(mu, L)
        return self.decode(z), input, mu, L

    def loss_function(self, args):
        recons = args[0]
        input = args[1]
        mu = args[2]
        L = args[3]

        recons_loss = F.mse_loss(recons, input)

        trace = (L ** 2).sum(dim=(-2, -1))

        log_det = 2 * torch.log(torch.diagonal(L, offset=0, dim1=-2, dim2=-1)).sum(-1)

        kld_loss = 0.5 * torch.mean(trace + mu.pow(2).sum(-1) - self.latent_dim - log_det)

        loss = recons_loss + self.kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':kld_loss.detach()}
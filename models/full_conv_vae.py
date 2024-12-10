import torch
from torch import nn
from torch.nn import functional as F


class FullConvVAE(nn.Module):
    def __init__(
        self,
        input_channels=3,
        image_size=32,
        latent_dim=10,
        hidden_dims=[8, 16, 32],
        fc_hidden_dims=[256, 128],
        kld_weight=1,
    ):
        super(FullConvVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.image_size = image_size
        self.hidden_dims = hidden_dims
        self.fc_hidden_dims = fc_hidden_dims
        self.kld_weight = kld_weight

        encoder_conv_layers = []
        in_channels = self.input_channels

        for h_dim in self.hidden_dims:
            encoder_conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder_conv = nn.Sequential(*encoder_conv_layers)

        self.conv_output_size = in_channels * 4 * 4

        fc_hidden_dims = self.fc_hidden_dims
        encoder_fc_layers = []
        in_features = self.conv_output_size
        for h_dim in fc_hidden_dims:
            encoder_fc_layers.append(
                nn.Sequential(nn.Linear(in_features, h_dim), nn.LeakyReLU())
            )
            in_features = h_dim

        self.encoder_fc = nn.Sequential(*encoder_fc_layers)

        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_m = nn.Linear(in_features, latent_dim)
        self.fc_M = nn.Linear(in_features, int(latent_dim * (latent_dim - 1) / 2))

        self.cov_idx_row, self.cov_idx_col = torch.tril_indices(latent_dim, latent_dim)
        non_diag = self.cov_idx_row != self.cov_idx_col
        self.cov_idx_col = self.cov_idx_col[non_diag]
        self.cov_idx_row = self.cov_idx_row[non_diag]

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        decoder_fc_layers = []
        in_features = latent_dim
        for h_dim in self.fc_hidden_dims[::-1]:
            decoder_fc_layers.append(
                nn.Sequential(nn.Linear(in_features, h_dim), nn.LeakyReLU())
            )
            in_features = h_dim

        self.decoder_fc = nn.Sequential(*decoder_fc_layers)

        self.decoder_fc_output = nn.Linear(in_features, self.conv_output_size)

        decoder_conv_layers = []
        hidden_dims_decoder = self.hidden_dims[::-1][1:]
        in_channels = self.hidden_dims[-1]
        for h_dim in hidden_dims_decoder:
            decoder_conv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels, h_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        decoder_conv_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, self.input_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.Sigmoid(),
            )
        )

        self.decoder_conv = nn.Sequential(*decoder_conv_layers)

    def encode(self, input):
        result = self.encoder_conv(input)
        result = torch.flatten(result, start_dim=1)

        result = self.encoder_fc(result)

        mu = self.fc_mu(result)
        m = self.fc_m(result)
        M = self.fc_M(result)

        return mu, m, M

    def decode(self, z):
        result = self.decoder_fc(z)
        result = self.decoder_fc_output(result)
        result = result.view(-1, 32, 4, 4)

        result = self.decoder_conv(result)
        return result

    def get_cov(self, m, M):
        batch_size = m.size(0)
        L = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=m.device)
        L[:, self.cov_idx_row, self.cov_idx_col] = M
        L += torch.diag_embed(torch.exp(m))
        return L

    def reparameterize(self, mu, L):
        eps = torch.randn((mu.size(0), self.latent_dim), device=mu.device)
        z = mu + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
        return z

    def forward(self, input):
        mu, m, M = self.encode(input)
        L = self.get_cov(m, M)
        z = self.reparameterize(mu, L)
        recon = self.decode(z)
        return recon, input, mu, L

    def reconstruct(self, input):
        mu, _, _ = self.encode(input)
        return self.decode(mu)

    def loss_function(self, args):
        recon = args[0]
        input = args[1]
        mu = args[2]
        L = args[3]

        recons_loss = F.mse_loss(recon, input, reduction="mean")

        trace = (L**2).sum(dim=(-2, -1))

        log_det = 2 * torch.log(torch.diagonal(L, offset=0, dim1=-2, dim2=-1)).sum(-1)

        kld_loss = 0.5 * torch.mean(
            trace + mu.pow(2).sum(-1) - self.latent_dim - log_det
        )

        loss = recons_loss + self.kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),
        }

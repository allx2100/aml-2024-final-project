import torch
from models.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from models.nvp import RealNVP


class NVPConvVAE(BaseVAE):
    def __init__(
        self,
        input_channels=3,
        image_size=32,
        latent_dim=10,
        hidden_dims=[8, 16, 32],
        fc_hidden_dims=[256, 128],
        kld_weight=1,
        num_flows=4,
        flow_hidden_dim=64,
    ):
        super(NVPConvVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.image_size = image_size
        self.hidden_dims = hidden_dims
        self.fc_hidden_dims = fc_hidden_dims
        self.kld_weight = kld_weight
        self.num_flows = num_flows

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
        self.fc_var = nn.Linear(in_features, latent_dim)

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

        self.flow = RealNVP(
            dim=latent_dim, hidden_dim=flow_hidden_dim, num_flows=num_flows
        )

    def encode(self, input):
        result = self.encoder_conv(input)
        result = torch.flatten(result, start_dim=1)

        result = self.encoder_fc(result)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z):
        result = self.decoder_fc(z)
        result = self.decoder_fc_output(result)
        result = result.view(-1, 32, 4, 4)

        result = self.decoder_conv(result)
        return result

    def reparameterize(self, mu, log_var):
        if log_var is None:  # For reconstruction
            z0 = mu
        else:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z0 = eps * std + mu
        zK, log_det_jacobian = self.flow(z0)
        zK = z0
        log_det_jacobian = torch.zeros(len(mu), device=mu.device)
        return z0, zK, log_det_jacobian

    def forward(self, input):
        mu, log_var = self.encode(input)
        z0, zK, log_det_jacobian = self.reparameterize(mu, log_var)
        return self.decode(zK), input, mu, log_var, z0, zK, log_det_jacobian

    def loss_function(self, args):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z0 = args[4]
        zK = args[5]
        log_det_jacobian = args[6]

        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)
            - log_det_jacobian
        )

        loss = recons_loss + self.kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),
        }

    def reconstruct(self, image):
        mu, _ = self.encode(image)
        _, zK, _ = self.reparameterize(mu, None)
        return self.decode(zK)

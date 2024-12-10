import torch
import torch.nn as nn


class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim, num_flows):
        super(RealNVP, self).__init__()
        self.dim = dim
        self.num_flows = num_flows

        self.masks = []
        self.s_t_networks = nn.ModuleList()
        for i in range(num_flows):
            mask = torch.zeros(dim)
            mask[i % 2 :: 2] = 1
            self.register_buffer("mask_%d" % i, mask)

            self.s_t_networks.append(
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dim * 2),
                )
            )

        self.masks = [getattr(self, "mask_%d" % i) for i in range(num_flows)]

    def forward(self, z):
        log_det_jacobian = torch.zeros(z.size(0), device=z.device)
        for mask, s_t_net in zip(self.masks, self.s_t_networks):
            mask = mask.to(z.device)
            z_masked = z * mask
            s_t = s_t_net(z_masked)
            s, t = s_t.chunk(2, dim=1)
            s = torch.tanh(s)

            z = z_masked + (1 - mask) * (z * torch.exp(s) + t)
            log_det_jacobian += ((1 - mask) * s).sum(dim=1)

        return z, log_det_jacobian

    def inverse(self, z):
        for mask, s_t_net in reversed(list(zip(self.masks, self.s_t_networks))):
            mask = mask.to(z.device)
            z_masked = z * mask
            s_t = s_t_net(z_masked)
            s, t = s_t.chunk(2, dim=1)
            s = torch.tanh(s)
            z = z_masked + (1 - mask) * ((z - t) * torch.exp(-s))

        return z

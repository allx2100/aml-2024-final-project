import torch
import torch.nn as nn
import torch.nn.functional as F


class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super(Invertible1x1Conv, self).__init__()

        w_init = torch.qr(torch.randn(num_channels, num_channels))[0]
        self.weight = nn.Parameter(w_init)

    def forward(self, x, reverse=False):
        if not reverse:
            z = x @ self.weight
            log_det = torch.slogdet(self.weight)[1] * torch.ones(
                x.size(0), device=x.device
            )

        else:
            w_inv = torch.inverse(self.weight)
            z = x @ w_inv
            log_det = -torch.slogdet(self.weight)[1] * torch.ones(
                x.size(0), device=x.device
            )

        return z, log_det


class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(AffineCoupling, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (dim - dim // 2) * 2),
        )

    def forward(self, x, reverse=False):
        d = x.size(1) // 2
        x1, x2 = x[:, :d], x[:, d:]

        st = self.net(x1)
        s, t = st[:, : x2.size(1)], st[:, x2.size(1) :]

        if not reverse:
            z2 = x2 * torch.exp(s) + t
            log_det = torch.sum(s, dim=1)
        else:
            z2 = (x2 - t) * torch.exp(-s)
            log_det = -torch.sum(s, dim=1)

        z = torch.cat([x1, z2], dim=1)
        return z, log_det


class Glow(nn.Module):
    def __init__(self, dim, hidden_dim, num_flows):
        super(Glow, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_flows = num_flows

        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(
                nn.ModuleList([Invertible1x1Conv(dim), AffineCoupling(dim, hidden_dim)])
            )

    def forward(self, z, reverse=False):
        log_det_total = torch.zeros(z.size(0), device=z.device)

        if not reverse:
            for conv, coupling in self.flows:
                z, log_det = conv(z, reverse=False)
                log_det_total += log_det
                z, log_det = coupling(z, reverse=False)
                log_det_total += log_det
        else:
            for conv, coupling in reversed(self.flows):
                z, log_det = coupling(z, reverse=True)
                log_det_total += log_det
                z, log_det = conv(z, reverse=True)
                log_det_total += log_det

        return z, log_det_total

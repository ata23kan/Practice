"""Fourier Neural Operator in 2D (Li et al., 2021).

An FNO maps an input field a(x) (with a few channels) to an output field u(x)
on the same grid:

    v_0     = P a                                  lift:    C_in -> width
    v_{l+1} = sigma( W_l v_l + K_l v_l + b_l )     L Fourier layers
    u       = Q v_L                                project: width -> C_out

P, W_l and Q act POINTWISE (a 1x1 convolution = the same small matrix at
every grid point). K_l is the only non-local part: a convolution
(K v)(x) = int kappa(x - y) v(y) dy, computed with the convolution theorem

    K v = F^{-1}( R . F(v) ),   R(k) in C^{width x width},   R(k) = 0 for |k| > modes.

R is learned DIRECTLY in Fourier space, and only the lowest `modes`
wavenumbers in each direction are kept. The parameters are attached to
wavenumbers, not to grid points, so the same network runs on any grid that
resolves those modes (the discretization invariance tested in run_*.py).

Tensor layout: (batch, channels, n_x, n_y), as for torch Conv2d.

No grid coordinates are fed in. Every piece above commutes with periodic
shifts, so the network is EXACTLY translation equivariant: shifting the input
shifts the output. This is the right bias for the periodic problems in this
folder. For position-dependent problems (fixed forcing, walls), append (x, y)
as extra input channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """K v = irfft2( R . rfft2(v) ), keeping |k_x| < modes and 0 <= k_y < modes.

    rfft2 stores only k_y >= 0 (the negative half follows from Hermitian
    symmetry because v is real), but k_x runs over both signs. The kept modes
    are therefore two corner blocks of the rfft2 array -- k_x >= 0 (top rows)
    and k_x < 0 (bottom rows) -- each with its own weights.
    """

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes, modes)
        self.weight_pos = nn.Parameter(scale * torch.randn(shape, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale * torch.randn(shape, dtype=torch.cfloat))

    def forward(self, v):
        batch, _, nx, ny = v.shape
        m = self.modes
        v_hat = torch.fft.rfft2(v)                                # (B, C, nx, ny//2+1)
        out_hat = torch.zeros(batch, self.out_channels, nx, ny // 2 + 1,
                              dtype=torch.cfloat, device=v.device)
        # Channel mixing, independently for every kept wavenumber:
        #   out_hat[b, o, k] = sum_i v_hat[b, i, k] R[i, o, k]
        out_hat[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", v_hat[:, :, :m, :m], self.weight_pos)
        out_hat[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", v_hat[:, :, -m:, :m], self.weight_neg)
        # rfft2 / irfft2 with the default normalization make R independent of
        # the grid size: a pure multiplier on the Fourier coefficients.
        return torch.fft.irfft2(out_hat, s=(nx, ny))


class FNO2d(nn.Module):
    """Lift -> n_layers Fourier layers -> project.

    homogeneous=True enforces G(c a_0, rest) = c G(a_0, rest) for c > 0 on
    input channel 0. Every operator in this folder is linear in the initial
    condition u0 (the heat equation is linear), so this is exact physics, not
    an approximation: the input u0 is scaled to unit RMS, and the output is
    scaled back. It matters when the network is applied repeatedly (see the
    semigroup trick in run_*.py): the second application gets a decayed,
    smaller u, which would otherwise be outside the training distribution.
    """

    def __init__(self, in_channels, out_channels=1, width=32, modes=12, n_layers=4,
                 projection_width=128, homogeneous=False):
        super().__init__()
        self.homogeneous = homogeneous
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes) for _ in range(n_layers))
        self.local = nn.ModuleList(nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers))
        self.project = nn.Sequential(
            nn.Conv2d(width, projection_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(projection_width, out_channels, kernel_size=1),
        )

    def forward(self, a):
        if self.homogeneous:
            scale = a[:, :1].pow(2).mean(dim=(2, 3), keepdim=True).sqrt()
            a = torch.cat([a[:, :1] / scale, a[:, 1:]], dim=1)

        v = self.lift(a)
        for l, (K, W) in enumerate(zip(self.spectral, self.local)):
            v = K(v) + W(v)          # global (Fourier) + local (pointwise) paths
            if l < len(self.spectral) - 1:
                v = F.gelu(v)
        u = self.project(v)

        if self.homogeneous:
            u = u * scale
        return u

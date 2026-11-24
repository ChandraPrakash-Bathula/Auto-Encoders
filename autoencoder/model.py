"""Deterministic convolutional Autoencoder (self-contained model file).

    x (1x128x128) --encoder--> z (latent_dim) --decoder--> x_hat (1x128x128)

The encoder downsamples with strided convolutions; the decoder upsamples with
nearest-neighbour interpolation + convolution (avoids checkerboard artefacts of
transposed convolutions). Normalisation: GroupNorm (batch-size independent).
The latent code is an unconstrained vector: there is NO prior on z, which is the
key structural difference to the VAE.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AEConfig:
    image_size: int = 128
    in_channels: int = 1
    latent_dim: int = 128
    base_channels: int = 32
    channel_mults: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])

    @property
    def n_stages(self) -> int:
        return len(self.channel_mults)

    @property
    def bottleneck_size(self) -> int:
        size = self.image_size // (2 ** self.n_stages)
        if size < 1 or self.image_size % (2 ** self.n_stages) != 0:
            raise ValueError("image_size must be divisible by 2**len(channel_mults)")
        return size


def conv_block(in_c: int, out_c: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, out_c),
        nn.SiLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        chans = [cfg.base_channels * m for m in cfg.channel_mults]
        layers: List[nn.Module] = [conv_block(cfg.in_channels, chans[0])]
        prev = chans[0]
        for c in chans:
            layers += [conv_block(prev, c, stride=2), conv_block(c, c)]
            prev = c
        self.features = nn.Sequential(*layers)
        self.flat_dim = prev * cfg.bottleneck_size ** 2
        self.to_latent = nn.Linear(self.flat_dim, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.features(x).flatten(1))


class Decoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        chans = [cfg.base_channels * m for m in cfg.channel_mults][::-1]
        self.bottleneck = (chans[0], cfg.bottleneck_size, cfg.bottleneck_size)
        self.from_latent = nn.Linear(cfg.latent_dim, chans[0] * cfg.bottleneck_size ** 2)
        layers: List[nn.Module] = []
        prev = chans[0]
        for c in chans:
            layers += [nn.Upsample(scale_factor=2, mode="nearest"), conv_block(prev, c), conv_block(c, c)]
            prev = c
        layers.append(nn.Conv2d(prev, cfg.in_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(z.shape[0], *self.bottleneck)
        return torch.sigmoid(self.net(h))  # images live in [0, 1]


class Autoencoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    # -- API ----------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 8) -> torch.Tensor:
        """Linear interpolation in latent space between two batches of images -> (steps, B, C, H, W)."""
        z1, z2 = self.encode(x1), self.encode(x2)
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        return torch.stack([self.decode((1 - a) * z1 + a * z2) for a in alphas])

    # -- loss ---------------------------------------------------------------
    @staticmethod
    def loss_function(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Mean-squared reconstruction error (per-pixel mean). Same recon term as the VAE."""
        return F.mse_loss(recon.float(), x.float())

    # -- persistence --------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(AEConfig(**ckpt["model_config"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt


def model_config_dict(cfg: AEConfig) -> dict:
    return asdict(cfg)

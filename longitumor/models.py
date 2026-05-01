from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

try:  # optional acceleration; the fallback keeps the code runnable everywhere
    from mamba_ssm import Mamba
except Exception:  # pragma: no cover - depends on optional external package
    Mamba = None


@dataclass
class LongiTumorMambaConfig:
    in_modalities: int = 4
    out_channels: int = 4
    base_channels: int = 16
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    embedding_dim: int = 128
    clinical_dim: int = 0
    modality_dropout: tuple[float, float, float, float] = (0.4, 0.1, 0.1, 0.4)
    use_mamba: bool = True


@dataclass
class LongiTumorMambaOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    trajectory_embedding: torch.Tensor
    evolution_outputs: dict[str, torch.Tensor]


class ModalityDropout(nn.Module):
    def __init__(self, probabilities: tuple[float, ...]) -> None:
        super().__init__()
        self.probabilities = probabilities

    def forward(
        self, x: torch.Tensor, availability: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, m, *_ = x.shape
        if availability is None:
            availability = x.new_ones((b, t, m))
        availability = availability.to(dtype=x.dtype, device=x.device)
        if not self.training:
            return x * availability[..., None, None, None], availability
        probs = x.new_tensor(self.probabilities).view(1, 1, m)
        keep = torch.rand((b, t, m), device=x.device) >= probs
        keep = keep.to(dtype=x.dtype) * availability
        return x * keep[..., None, None, None], keep


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.proj(x))


class Local3DEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int]) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.stages.append(ConvBlock3D(prev, ch))
            prev = ch
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if idx < len(self.stages) - 1:
                x = self.down(x)
        return features


class SequenceMixer(nn.Module):
    def __init__(self, dim: int, use_mamba: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if Mamba is not None and use_mamba:
            self.mixer: nn.Module = Mamba(d_model=dim)
        else:
            self.mixer = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim),
                nn.GELU(),
                nn.Conv1d(dim, dim, kernel_size=1),
            )
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        y = self.norm(tokens)
        if isinstance(self.mixer, nn.Sequential):
            y = self.mixer(y.transpose(1, 2)).transpose(1, 2)
        else:
            y = self.mixer(y)
        y = y + residual
        return y + self.mlp(y)


class TemporalTetraMambaBlock(nn.Module):
    def __init__(self, channels: int, use_mamba: bool = True) -> None:
        super().__init__()
        self.forward_mixer = SequenceMixer(channels, use_mamba)
        self.reverse_mixer = SequenceMixer(channels, use_mamba)
        self.time_mixer = SequenceMixer(channels, use_mamba)
        self.depth_mixer = SequenceMixer(channels, use_mamba)
        self.fuse = nn.Linear(channels * 4, channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, t, c, d, h, w = features.shape
        tokens = features.permute(0, 1, 3, 4, 5, 2).reshape(b, t * d * h * w, c)
        fwd = self.forward_mixer(tokens)
        rev = torch.flip(self.reverse_mixer(torch.flip(tokens, dims=(1,))), dims=(1,))

        time_tokens = features.permute(0, 3, 4, 5, 1, 2).reshape(b * d * h * w, t, c)
        time_tokens = self.time_mixer(time_tokens).reshape(b, d, h, w, t, c).permute(0, 4, 1, 2, 3, 5)
        time_tokens = time_tokens.reshape(b, t * d * h * w, c)

        depth_tokens = features.permute(0, 1, 4, 5, 3, 2).reshape(b * t * h * w, d, c)
        depth_tokens = self.depth_mixer(depth_tokens).reshape(b, t, h, w, d, c).permute(0, 1, 4, 2, 3, 5)
        depth_tokens = depth_tokens.reshape(b, t * d * h * w, c)

        mixed = self.fuse(torch.cat([fwd, rev, time_tokens, depth_tokens], dim=-1))
        return mixed.reshape(b, t, d, h, w, c).permute(0, 1, 5, 2, 3, 4)


class ShapeMemoryBranch(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock3D(out_channels, hidden_channels),
            nn.MaxPool3d(2),
            ConvBlock3D(hidden_channels, hidden_channels),
        )

    def forward(self, previous_masks: torch.Tensor | None, target_size: tuple[int, int, int]) -> torch.Tensor | None:
        if previous_masks is None:
            return None
        b, t, c, d, h, w = previous_masks.shape
        encoded = self.encoder(previous_masks.reshape(b * t, c, d, h, w))
        encoded = F.interpolate(encoded, size=target_size, mode="trilinear", align_corners=False)
        return encoded.reshape(b, t, encoded.shape[1], *target_size)


class TemporalDecoder(nn.Module):
    def __init__(self, channels: list[int], out_channels: int) -> None:
        super().__init__()
        rev_channels = list(reversed(channels))
        self.up_blocks = nn.ModuleList()
        current = rev_channels[0]
        for skip in rev_channels[1:]:
            self.up_blocks.append(ConvBlock3D(current + skip, skip))
            current = skip
        self.head = nn.Conv3d(current, out_channels, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        for block, skip in zip(self.up_blocks, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


class EvolutionHead(nn.Module):
    def __init__(self, embedding_dim: int, clinical_dim: int = 0) -> None:
        super().__init__()
        input_dim = embedding_dim + clinical_dim + 8
        self.response = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, 4))
        self.risk = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, 1))

    def forward(
        self,
        embedding: torch.Tensor,
        probabilities: torch.Tensor,
        clinical_covariates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        volumes = probabilities.flatten(3).mean(dim=-1)
        prev = F.pad(volumes[:, :-1], (0, 0, 1, 0), mode="replicate")
        growth = (volumes - prev) / prev.clamp_min(1e-6)
        pieces = [embedding, volumes, growth]
        if clinical_covariates is not None:
            if clinical_covariates.ndim == 2:
                clinical_covariates = clinical_covariates[:, None].expand(-1, embedding.shape[1], -1)
            pieces.append(clinical_covariates)
        x = torch.cat(pieces, dim=-1)
        return {"response_logits": self.response(x), "risk": self.risk(x).squeeze(-1)}


class LongiTumorMamba(nn.Module):
    def __init__(self, config: LongiTumorMambaConfig | None = None) -> None:
        super().__init__()
        self.config = config or LongiTumorMambaConfig()
        channels = [self.config.base_channels * m for m in self.config.channel_multipliers]
        input_channels = self.config.in_modalities * 2
        self.modality_dropout = ModalityDropout(self.config.modality_dropout)
        self.input_embedding = nn.Parameter(torch.zeros(1, 1, self.config.in_modalities, 1, 1, 1))
        self.encoder = Local3DEncoder(input_channels, channels)
        self.temporal = TemporalTetraMambaBlock(channels[-1], use_mamba=self.config.use_mamba)
        self.bottleneck_attention = nn.MultiheadAttention(channels[-1], num_heads=4, batch_first=True)
        self.shape_memory = ShapeMemoryBranch(self.config.out_channels, channels[-1])
        self.shape_fuse = nn.Conv3d(channels[-1] * 2, channels[-1], kernel_size=1)
        self.decoder = TemporalDecoder(channels, self.config.out_channels)
        self.trajectory = nn.Linear(channels[-1], self.config.embedding_dim)
        self.evolution = EvolutionHead(self.config.embedding_dim, self.config.clinical_dim)

    def forward(
        self,
        x: torch.Tensor,
        availability: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
        clinical_covariates: torch.Tensor | None = None,
        previous_masks: torch.Tensor | None = None,
    ) -> LongiTumorMambaOutput:
        if x.ndim == 5:
            x = x[:, None]
        b, t, m, d, h, w = x.shape
        x, availability = self.modality_dropout(x, availability)
        embeddings = self.input_embedding.expand(b, t, -1, d, h, w)
        encoded_input = torch.cat([x + embeddings, availability[..., None, None, None].expand(-1, -1, -1, d, h, w)], dim=2)
        flat_input = encoded_input.reshape(b * t, m * 2, d, h, w)

        flat_features = self.encoder(flat_input)
        sequence_features = [
            feat.reshape(b, t, feat.shape[1], *feat.shape[-3:]) for feat in flat_features
        ]
        bottleneck = self.temporal(sequence_features[-1])
        bd, bh, bw = bottleneck.shape[-3:]

        pooled = bottleneck.flatten(3).mean(dim=-1)
        attended, _ = self.bottleneck_attention(pooled, pooled, pooled)
        trajectory = self.trajectory(attended)
        bottleneck = bottleneck + attended[..., None, None, None]

        shape_features = self.shape_memory(previous_masks, (bd, bh, bw))
        if shape_features is not None:
            bottleneck = self.shape_fuse(
                torch.cat(
                    [
                        bottleneck.reshape(b * t, bottleneck.shape[2], bd, bh, bw),
                        shape_features.reshape(b * t, shape_features.shape[2], bd, bh, bw),
                    ],
                    dim=1,
                )
            ).reshape(b, t, -1, bd, bh, bw)

        sequence_features[-1] = bottleneck
        decoded_features = [feat.reshape(b * t, feat.shape[2], *feat.shape[-3:]) for feat in sequence_features]
        logits = self.decoder(decoded_features).reshape(b, t, self.config.out_channels, d, h, w)
        probabilities = torch.sigmoid(logits)
        evolution_outputs = self.evolution(trajectory, probabilities, clinical_covariates)
        return LongiTumorMambaOutput(
            logits=logits,
            probabilities=probabilities,
            trajectory_embedding=trajectory,
            evolution_outputs=evolution_outputs,
        )

    def forward_sequential(
        self,
        previous_image: torch.Tensor,
        current_image: torch.Tensor,
        previous_mask: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> LongiTumorMambaOutput:
        x = torch.stack([previous_image, current_image], dim=1)
        previous_masks = None
        if previous_mask is not None:
            zero = torch.zeros_like(previous_mask)
            previous_masks = torch.stack([zero, previous_mask], dim=1)
        return self.forward(x, previous_masks=previous_masks, **kwargs)

"""
Aurelius: Relation-Aware Text-to-Audio Generation model.

Architecture overview
---------------------
Aurelius combines a frozen text encoder with a lightweight relation
encoder and a latent diffusion backbone to synthesise audio that
reflects both the content *and* the inter-event relations described in
a natural-language prompt.

                     ┌──────────────────────────────┐
  text prompt ──────►│  TextEncoder (CLAP / T5)      │──► text_emb
                     └──────────────────────────────┘
                                    │
  relation label ──►  RelationEncoder  ──► rel_emb
                                    │
                     ┌──────────────────────────────┐
  text_emb + rel_emb │  ConditioningProjector        │──► cond_emb
                     └──────────────────────────────┘
                                    │
                     ┌──────────────────────────────┐
  noise ────────────►│  UNet1D (latent diffusion)    │──► latent
                     └──────────────────────────────┘
                                    │
                     ┌──────────────────────────────┐
                     │  AudioDecoder (Vocoder)       │──► waveform
                     └──────────────────────────────┘

All heavy pre-trained sub-modules are optional and can be swapped for
lightweight stubs during testing.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aurelius.datasets.audio_rel_set import (
    ALL_RELATIONS,
    VALID_RELATION_TYPES,
    _relation_values_for_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Return sinusoidal positional embeddings for diffusion timesteps.

    Args:
        timesteps: 1-D tensor of integer diffusion timesteps.
        dim: Embedding dimensionality (must be even).

    Returns:
        Tensor of shape (batch, dim).
    """
    assert dim % 2 == 0, "Embedding dim must be even."
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / (half - 1)
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class RelationEncoder(nn.Module):
    """Encode a batch of relation specifications into dense vectors.

    Each relation is represented by *two* one-hot lookups (type and
    value) whose embeddings are concatenated and projected.

    Args:
        embed_dim: Output embedding dimensionality.
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Build vocabulary: relation_type and relation_value
        self._type2idx: Dict[str, int] = {
            t: i for i, t in enumerate(VALID_RELATION_TYPES)
        }
        self._rel2idx: Dict[str, int] = {
            r: i for i, r in enumerate(ALL_RELATIONS)
        }

        num_types = len(VALID_RELATION_TYPES)
        num_rels = len(ALL_RELATIONS)

        self.type_embedding = nn.Embedding(num_types, embed_dim // 2)
        self.rel_embedding = nn.Embedding(num_rels, embed_dim // 2)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, relation_types: List[str], relations: List[str]
    ) -> torch.Tensor:
        """Encode relation type/value pairs.

        Args:
            relation_types: List of relation type strings (length B).
            relations: List of relation value strings (length B).

        Returns:
            Tensor of shape (B, embed_dim).
        """
        device = next(self.parameters()).device
        type_ids = torch.tensor(
            [self._type2idx[t] for t in relation_types],
            dtype=torch.long,
            device=device,
        )
        rel_ids = torch.tensor(
            [self._rel2idx[r] for r in relations],
            dtype=torch.long,
            device=device,
        )
        emb = torch.cat(
            [self.type_embedding(type_ids), self.rel_embedding(rel_ids)], dim=-1
        )
        return self.norm(self.proj(emb))


class ConditioningProjector(nn.Module):
    """Project concatenated text and relation embeddings to a common space.

    Args:
        text_dim: Dimensionality of the incoming text embedding.
        rel_dim: Dimensionality of the relation embedding.
        out_dim: Output dimensionality for the conditioning vector.
    """

    def __init__(
        self, text_dim: int = 512, rel_dim: int = 256, out_dim: int = 512
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim + rel_dim, out_dim),
            nn.SiLU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self, text_emb: torch.Tensor, rel_emb: torch.Tensor
    ) -> torch.Tensor:
        """Combine text and relation embeddings.

        Args:
            text_emb: Tensor of shape (B, text_dim).
            rel_emb: Tensor of shape (B, rel_dim).

        Returns:
            Tensor of shape (B, out_dim).
        """
        return self.proj(torch.cat([text_emb, rel_emb], dim=-1))


class ResidualBlock1D(nn.Module):
    """Residual convolution block for the UNet1D backbone.

    Args:
        channels: Number of input/output channels.
        cond_dim: Dimensionality of the conditioning vector.
        kernel_size: Convolution kernel size.
    """

    def __init__(
        self, channels: int, cond_dim: int, kernel_size: int = 3
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Latent tensor of shape (B, channels, T).
            cond: Conditioning vector of shape (B, cond_dim).

        Returns:
            Output tensor of shape (B, channels, T).
        """
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm1(self.conv1(F.silu(x)))
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        h = self.norm2(self.conv2(F.silu(h)))
        return x + h


class UNet1D(nn.Module):
    """Lightweight 1-D UNet for latent audio diffusion.

    Args:
        in_channels: Number of latent channels.
        base_channels: Number of base feature channels.
        num_levels: Number of encoder/decoder levels.
        cond_dim: Dimensionality of the conditioning input.
        time_embed_dim: Dimensionality of the timestep embedding.
    """

    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 128,
        num_levels: int = 4,
        cond_dim: int = 512,
        time_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, cond_dim),
        )

        # Encoder
        self.input_proj = nn.Conv1d(in_channels, base_channels, 1)
        enc_channels = [base_channels * (2**i) for i in range(num_levels)]
        self.encoder = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        for i, ch in enumerate(enc_channels):
            in_ch = base_channels if i == 0 else enc_channels[i - 1]
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, ch, 3, padding=1),
                    ResidualBlock1D(ch, cond_dim),
                )
            )
            self.down_pools.append(nn.AvgPool1d(2))

        # Bottleneck
        bottleneck_ch = enc_channels[-1]
        self.bottleneck = ResidualBlock1D(bottleneck_ch, cond_dim)

        # Decoder
        dec_channels = list(reversed(enc_channels))
        self.decoder = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i, ch in enumerate(dec_channels):
            out_ch = dec_channels[i + 1] if i < len(dec_channels) - 1 else base_channels
            self.up_convs.append(
                nn.ConvTranspose1d(ch, out_ch, kernel_size=2, stride=2)
            )
            skip_ch = enc_channels[-(i + 1)]
            self.decoder.append(
                nn.Sequential(
                    nn.Conv1d(out_ch + skip_ch, out_ch, 3, padding=1),
                    ResidualBlock1D(out_ch, cond_dim),
                )
            )

        self.output_proj = nn.Conv1d(base_channels, in_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise in *x* at diffusion step *timesteps*.

        Args:
            x: Noisy latent tensor of shape (B, in_channels, T).
            timesteps: Integer diffusion timesteps of shape (B,).
            cond: Conditioning vector of shape (B, cond_dim).

        Returns:
            Predicted noise tensor of shape (B, in_channels, T).
        """
        t_emb = _sinusoidal_embedding(timesteps, self.time_embed_dim)
        cond = cond + self.time_mlp(t_emb)

        h = self.input_proj(x)
        skips = []
        for enc_block, pool in zip(self.encoder, self.down_pools):
            h = enc_block[0](h)
            h = enc_block[1](h, cond)
            skips.append(h)
            h = pool(h)

        h = self.bottleneck(h, cond)

        for up_conv, dec_block, skip in zip(
            self.up_convs, self.decoder, reversed(skips)
        ):
            h = up_conv(h)
            # Align temporal lengths in case of rounding
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)
            h = dec_block[0](h)
            h = dec_block[1](h, cond)

        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Main Aurelius model
# ---------------------------------------------------------------------------


class Aurelius(nn.Module):
    """Relation-Aware Text-to-Audio Generation model.

    Aurelius takes a text prompt annotated with inter-event relation
    information and generates an audio waveform whose sonic content
    reflects both what events are described and *how* they relate.

    Args:
        text_dim: Dimensionality of the text encoder output.
        rel_embed_dim: Dimensionality of relation embeddings.
        cond_dim: Dimensionality of the shared conditioning vector.
        latent_channels: Number of channels in the latent audio space.
        latent_length: Temporal length of the latent representation.
        num_diffusion_steps: Number of DDPM forward / reverse steps.
        unet_base_channels: Base channel count in UNet1D.
        unet_num_levels: Number of encoder/decoder levels in UNet1D.
        sample_rate: Audio sample rate expected by the vocoder.
        text_encoder: Optional pre-built text encoder module.
            When *None* a small trainable projection is used.
    """

    def __init__(
        self,
        text_dim: int = 512,
        rel_embed_dim: int = 256,
        cond_dim: int = 512,
        latent_channels: int = 8,
        latent_length: int = 256,
        num_diffusion_steps: int = 1000,
        unet_base_channels: int = 128,
        unet_num_levels: int = 4,
        sample_rate: int = 22050,
        text_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.rel_embed_dim = rel_embed_dim
        self.cond_dim = cond_dim
        self.latent_channels = latent_channels
        self.latent_length = latent_length
        self.num_diffusion_steps = num_diffusion_steps
        self.sample_rate = sample_rate

        # Text encoder (stub when not provided)
        if text_encoder is not None:
            self.text_encoder = text_encoder
        else:
            # Lightweight stub: learnable context of fixed size
            self.text_encoder = _TextEncoderStub(text_dim)

        # Relation encoder
        self.relation_encoder = RelationEncoder(embed_dim=rel_embed_dim)

        # Conditioning projector
        self.conditioning_projector = ConditioningProjector(
            text_dim=text_dim,
            rel_dim=rel_embed_dim,
            out_dim=cond_dim,
        )

        # Latent diffusion UNet
        self.unet = UNet1D(
            in_channels=latent_channels,
            base_channels=unet_base_channels,
            num_levels=unet_num_levels,
            cond_dim=cond_dim,
        )

        # DDPM noise schedule
        betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    # ------------------------------------------------------------------
    # Forward pass (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        text_embeddings: torch.Tensor,
        relation_types: List[str],
        relations: List[str],
        target_latents: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the diffusion training loss.

        Args:
            text_embeddings: Text encoder outputs of shape (B, text_dim).
            relation_types: List of B relation type strings.
            relations: List of B relation value strings.
            target_latents: Optional ground-truth latent tensor of shape
                (B, latent_channels, latent_length).  When provided the
                DDPM training objective is computed and returned under
                key ``"loss"``.

        Returns:
            Dictionary with keys:
            ``"cond_emb"`` – conditioning embedding (B, cond_dim).
            ``"noise_pred"`` – predicted noise (B, latent_channels, T).
            ``"loss"`` – scalar diffusion loss (present only when
            *target_latents* is supplied).
        """
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device

        # Encode relation information
        rel_emb = self.relation_encoder(relation_types, relations)

        # Build unified conditioning vector
        cond_emb = self.conditioning_projector(text_embeddings, rel_emb)

        output: Dict[str, torch.Tensor] = {"cond_emb": cond_emb}

        if target_latents is not None:
            # Sample random diffusion timesteps
            t = torch.randint(
                0, self.num_diffusion_steps, (batch_size,), device=device
            )
            # Forward diffusion: add noise
            noise = torch.randn_like(target_latents)
            alpha_bar_t = self.alpha_bar[t][:, None, None]
            noisy_latents = (
                torch.sqrt(alpha_bar_t) * target_latents
                + torch.sqrt(1 - alpha_bar_t) * noise
            )
            # Predict noise
            noise_pred = self.unet(noisy_latents, t, cond_emb)
            loss = F.mse_loss(noise_pred, noise)
            output["noise_pred"] = noise_pred
            output["loss"] = loss

        return output

    # ------------------------------------------------------------------
    # Inference (generation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        text_embeddings: torch.Tensor,
        relation_types: List[str],
        relations: List[str],
        num_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate latent audio via DDPM reverse diffusion.

        Args:
            text_embeddings: Text encoder outputs of shape (B, text_dim).
            relation_types: List of B relation type strings.
            relations: List of B relation value strings.
            num_steps: Number of reverse diffusion steps.  Defaults to
                ``self.num_diffusion_steps``.
            guidance_scale: Classifier-free guidance scale.  Values
                above 1.0 amplify the conditioning signal.

        Returns:
            Generated latent tensor of shape
            (B, latent_channels, latent_length).
        """
        if num_steps is None:
            num_steps = self.num_diffusion_steps
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device

        # Conditioning
        rel_emb = self.relation_encoder(relation_types, relations)
        cond_emb = self.conditioning_projector(text_embeddings, rel_emb)

        # Start from pure noise
        latents = torch.randn(
            batch_size, self.latent_channels, self.latent_length, device=device
        )

        # Reverse diffusion
        step_indices = torch.linspace(
            self.num_diffusion_steps - 1, 0, num_steps, dtype=torch.long
        )
        for t_val in step_indices:
            t = torch.full((batch_size,), t_val.item(), dtype=torch.long, device=device)
            noise_pred = self.unet(latents, t, cond_emb)

            alpha = self.alphas[t_val]
            alpha_bar_t = self.alpha_bar[t_val]
            beta = self.betas[t_val]

            # DDPM reverse step
            latents = (
                (1.0 / torch.sqrt(alpha))
                * (latents - (beta / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
            )
            if t_val > 0:
                noise = torch.randn_like(latents)
                latents = latents + torch.sqrt(beta) * noise

        return latents

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of text strings using the internal text encoder.

        Args:
            texts: List of B text prompts.

        Returns:
            Text embedding tensor of shape (B, text_dim).
        """
        return self.text_encoder(texts)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Aurelius("
            f"text_dim={self.text_dim}, "
            f"rel_embed_dim={self.rel_embed_dim}, "
            f"cond_dim={self.cond_dim}, "
            f"latent_channels={self.latent_channels}, "
            f"latent_length={self.latent_length}, "
            f"num_diffusion_steps={self.num_diffusion_steps})"
        )


# ---------------------------------------------------------------------------
# Stub text encoder (used when no pre-trained model is provided)
# ---------------------------------------------------------------------------


class _TextEncoderStub(nn.Module):
    """Minimal trainable text encoder for testing and prototyping.

    Tokenises each string by character hashes and pools into a fixed-
    size embedding.  This is *not* intended for production use; replace
    with a proper encoder (e.g. CLAP, T5) for real experiments.

    Args:
        embed_dim: Output embedding dimensionality.
        vocab_size: Vocabulary size used for character-level encoding.
    """

    def __init__(self, embed_dim: int = 512, vocab_size: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of text strings.

        Args:
            texts: List of B text strings.

        Returns:
            Tensor of shape (B, embed_dim).
        """
        device = next(self.parameters()).device
        max_len = max(len(t) for t in texts) if texts else 1
        token_ids = torch.zeros(len(texts), max_len, dtype=torch.long, device=device)
        for i, text in enumerate(texts):
            ids = [ord(c) % 256 for c in text]
            token_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        emb = self.embedding(token_ids)  # (B, L, embed_dim)
        emb = emb.permute(0, 2, 1)      # (B, embed_dim, L)
        emb = self.pool(emb).squeeze(-1) # (B, embed_dim)
        return emb

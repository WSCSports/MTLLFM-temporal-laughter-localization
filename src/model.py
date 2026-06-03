"""
MTLLFM model definition: Adaptive Multimodal Fusion for temporal laughter
localization.

The model fuses frozen audio (HuBERT) and visual (MAE) features with:
  1. Modality-specific projection to a shared hidden space.
  2. Temporal Softmax Pooling per modality (learned attention over timesteps).
  3. Adaptive Modality Gating (complementary softmax weights).
  4. A linear classification head.

The per-timestep attention distributions are returned alongside the logits and
are used at inference time for post-hoc temporal localization (see localize.py).
"""
from typing import Any, Dict, Tuple

import torch
from torch import nn


class SoftmaxPooling(nn.Module):
    """Softmax-weighted temporal pooling that emphasizes salient timesteps."""

    def __init__(self, input_dim: int, softmax_tanh: bool = True):
        super().__init__()
        if softmax_tanh:
            self.attention = nn.Sequential(nn.Linear(input_dim, 1), nn.Tanh())
        else:
            self.attention = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """x: (B, T, D) -> pooled (B, D) and attention scores (B, T)."""
        attn_scores = torch.softmax(self.attention(x), dim=1)  # over time
        pooled = (attn_scores * x).sum(dim=1)
        return pooled, {"attn_scores": attn_scores.squeeze(-1).detach().cpu().numpy()}


class GatedFusion(nn.Module):
    """Adaptive modality gating with complementary (softmax) weights."""

    def __init__(self, hidden_dim: int, gating_type: str = "softmax"):
        super().__init__()
        self.gating_type = gating_type
        self.audio_gate = nn.Linear(hidden_dim, 1)
        self.vision_gate = nn.Linear(hidden_dim, 1)
        if gating_type == "sigmoid":
            self.activation = nn.Sigmoid()
        elif gating_type != "softmax":
            raise ValueError("gating_type must be 'sigmoid' or 'softmax'.")

    def forward(self, audio_features: torch.Tensor, vision_features: torch.Tensor):
        if self.gating_type == "sigmoid":
            w_audio = self.activation(self.audio_gate(audio_features))
            w_vision = self.activation(self.vision_gate(vision_features))
        else:  # softmax: enforce w_audio + w_vision = 1
            gate_logits = torch.cat(
                (self.audio_gate(audio_features), self.vision_gate(vision_features)),
                dim=1,
            )
            weights = torch.softmax(gate_logits, dim=1)
            w_audio, w_vision = weights[:, 0:1], weights[:, 1:2]
        fused = w_audio * audio_features + w_vision * vision_features
        gate_weights = torch.cat((w_audio, w_vision), dim=1).detach().cpu().numpy()
        return fused, gate_weights


class AudioVisionSoftmaxPool(nn.Module):
    """MTLLFM: audio-vision model with temporal softmax pooling + gated fusion."""

    def __init__(
        self,
        audio_dim: int = 1024,
        vision_dim: int = 768,
        hidden_dim: int = 1024,
        output_dim: int = 2,
        dropout: float = 0.5,
        fusion_type: str = "gated",
        activation_and_norm: bool = False,
        gating_type: str = "softmax",
        softmax_tanh: bool = True,
    ):
        super().__init__()
        self.fusion_type = fusion_type

        if activation_and_norm:
            self.audio_proj = nn.Sequential(
                nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
            )
            self.vision_proj = nn.Sequential(
                nn.Linear(vision_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
            )
        else:
            self.audio_proj = nn.Linear(audio_dim, hidden_dim)
            self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        self.audio_pooling = SoftmaxPooling(hidden_dim, softmax_tanh)
        self.vision_pooling = SoftmaxPooling(hidden_dim, softmax_tanh)

        if fusion_type == "gated":
            self.fusion_layer = GatedFusion(hidden_dim, gating_type=gating_type)
            self.fusion_fc = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_type == "concat":
            self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            raise ValueError("fusion_type must be 'gated' or 'concat'.")

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, audio: torch.Tensor, vision: torch.Tensor, is_training: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        audio:  (B, T_audio, audio_dim)   HuBERT embeddings
        vision: (B, T_vision, vision_dim) MAE embeddings
        Returns logits (B, output_dim) and a dict of attention weights.
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if vision.dim() == 2:
            vision = vision.unsqueeze(1)

        audio = self.audio_proj(audio)
        vision = self.vision_proj(vision)

        audio_pooled, audio_att = self.audio_pooling(audio)
        vision_pooled, vision_att = self.vision_pooling(vision)

        if self.fusion_type == "gated":
            fused, gates_att = self.fusion_layer(audio_pooled, vision_pooled)
            fused = self.fusion_fc(fused)
        else:
            fused = self.fusion_fc(torch.cat([audio_pooled, vision_pooled], dim=1))
            gates_att = None

        att_weights = {
            "audio_att": audio_att["attn_scores"],
            "vision_att": vision_att["attn_scores"],
            "gates_att": gates_att,
        }

        if is_training:
            fused = self.dropout(fused)
        return self.fc(fused), att_weights

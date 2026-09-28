"""GazeTrajectoryPredictor — GazeMani gaze predictor.

  encoder_kind = "clip_clip"             (frozen CLIP-B/16 vision + text)
  head_kind    = "prompt_then_temporal_224"  (K=3 prompt-then-temporal +
                                              14->224 spatial upsampler)

Inference contract:
    forward(vis_features, text_features, text_mask=None) -> logits [B, n_anchors, 224, 224]
"""
from __future__ import annotations


import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Frozen encoders (CLIP-B/16 vision + text).
# ---------------------------------------------------------------------------

class _ClipVisionEncoder(nn.Module):
    def __init__(self, model_id, fusion_dim):
        super().__init__()
        from transformers import CLIPModel
        clip = CLIPModel.from_pretrained(model_id)
        self.vision = clip.vision_model
        for p in self.vision.parameters():
            p.requires_grad = False
        VIS_DIM = self.vision.config.hidden_size
        self.proj = nn.Linear(VIS_DIM, fusion_dim)

    @torch.no_grad()
    def _raw(self, images):
        return self.vision(pixel_values=images).last_hidden_state[:, 1:, :]

    def forward(self, images):
        return self.proj(self._raw(images))


class _ClipTextEncoder(nn.Module):
    def __init__(self, model_id, fusion_dim):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer
        clip = CLIPModel.from_pretrained(model_id)
        self.text = clip.text_model
        for p in self.text.parameters():
            p.requires_grad = False
        TXT_DIM = self.text.config.hidden_size
        self.proj = nn.Linear(TXT_DIM, fusion_dim)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)

    @torch.no_grad()
    def _raw(self, prompts, device):
        toks = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        toks = {k: v.to(device) for k, v in toks.items()}
        out = self.text(**toks).last_hidden_state
        return out, toks["attention_mask"]

    def forward(self, prompts):
        device = next(self.parameters()).device
        raw, mask = self._raw(prompts, device)
        return self.proj(raw), mask


# ---------------------------------------------------------------------------
# Prompt-then-Temporal head (K-frame; per-frame prompt cross-attn -> temporal
# fusion -> readout).
# ---------------------------------------------------------------------------

class _PromptThenTemporalHead(nn.Module):
    """Per-frame prompt conditioning BEFORE temporal fusion.

    Order of operations:
        1. Each of the K vision frames cross-attends to the text features
           (prompt is injected per frame, upstream of any temporal mixing).
        2. The K prompt-conditioned token grids are fused via a small temporal
           transformer over (K * P) tokens.
        3. The last (current) frame's contextualised tokens drive a final
           query-based readout that emits per-position logits over the P=G*G
           tokens.

    Inputs:
        vis_features_seq: [B, K, P, D]   (K-frame stack of projected vision tokens)
        text_features:    [B, T, D]
        text_mask:        [B, T] bool (True = real token)

    Output:
        logits: [B, n_anchors, G, G]

    ``_PromptThenTemporalHead224`` reuses stages 1-2 (prompt cross-attn +
    temporal fusion) and replaces stage 3 with a 14->224 spatial upsampler.
    """
    def __init__(self, fusion_dim: int, n_anchors: int, grid: int,
                 K: int = 3, n_temporal_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.n_anchors = n_anchors
        self.grid = grid
        self.K = K
        # 1) Per-frame prompt conditioning via cross-attention (vision queries
        #    text). Pre-norm style: norm before each block, residual after.
        self.norm_q_pc = nn.LayerNorm(fusion_dim)
        self.norm_kv_pc = nn.LayerNorm(fusion_dim)
        self.vis_to_text_attn = nn.MultiheadAttention(
            fusion_dim, num_heads=n_heads, batch_first=True)
        self.norm_ffn_pc = nn.LayerNorm(fusion_dim)
        self.ffn_pc = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )
        # 2) Temporal fusion over (K * P) prompt-conditioned tokens.
        self.temporal_pos = nn.Parameter(torch.randn(K, fusion_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=n_heads,
            dim_feedforward=fusion_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_temporal_layers)
        # 3) Final readout: n_anchors learnable queries that cross-attend to
        #    the current (last) frame's contextualised tokens, then dot-product
        #    with the same tokens to produce P logits per anchor.
        self.time_queries = nn.Parameter(torch.randn(n_anchors, fusion_dim) * 0.02)
        self.norm_q_final = nn.LayerNorm(fusion_dim)
        self.norm_kv_final = nn.LayerNorm(fusion_dim)
        self.final_attn = nn.MultiheadAttention(
            fusion_dim, num_heads=n_heads, batch_first=True)

    def forward(self, vis_features_seq, text_features, text_mask=None):
        """vis_features_seq: [B, K, P, D] -> heatmap logits for the LAST frame.

        Falls back when caller passes [B, P, D]: treat as K=1, skip the
        temporal encoder, but still apply prompt conditioning + final readout.
        """
        if vis_features_seq.dim() == 3:
            vis_features_seq = vis_features_seq.unsqueeze(1)
            single_frame = True
        else:
            single_frame = False
        B, K, P, D = vis_features_seq.shape
        if not single_frame and K != self.K:
            raise ValueError(
                f"_PromptThenTemporalHead expects K={self.K} frames, got K={K}")
        T = text_features.size(1)

        # ----- 1) Per-frame: vision tokens cross-attend to text tokens -----
        vis_flat = vis_features_seq.reshape(B * K, P, D)
        text_expanded = text_features.unsqueeze(1).expand(B, K, T, D)\
                                     .reshape(B * K, T, D)
        if text_mask is not None:
            tm_bool = text_mask.bool()
            key_pad = ~tm_bool.unsqueeze(1).expand(B, K, T).reshape(B * K, T)
        else:
            key_pad = None
        q = self.norm_q_pc(vis_flat)
        kv = self.norm_kv_pc(text_expanded)
        attn_out, _ = self.vis_to_text_attn(q, kv, kv, key_padding_mask=key_pad)
        x = vis_flat + attn_out
        x = x + self.ffn_pc(self.norm_ffn_pc(x))
        x = x.reshape(B, K, P, D)

        # ----- 2) Temporal fusion (skipped in single-frame fallback) -----
        if not single_frame:
            x = x + self.temporal_pos.view(1, K, 1, D)
            x = x.reshape(B, K * P, D)
            x = self.temporal_encoder(x)
            x = x.reshape(B, K, P, D)
        current = x[:, -1]                                                    # [B, P, D]

        # ----- 3) Final readout -----
        queries = self.time_queries.unsqueeze(0).expand(B, -1, -1)            # [B, N, D]
        q_f = self.norm_q_final(queries)
        kv_f = self.norm_kv_final(current)
        anchor_feats, _ = self.final_attn(q_f, kv_f, kv_f)                    # [B, N, D]
        logits = torch.einsum("bnd,bpd->bnp", anchor_feats, current)          # [B, N, P]
        G = self.grid
        return logits.reshape(B, self.n_anchors, G, G)


class _PromptThenTemporalHead224(_PromptThenTemporalHead):
    """``_PromptThenTemporalHead`` with the per-position readout replaced by a
    14->224 spatial upsampler.

    Up to step (2) (per-frame prompt cross-attn + temporal fusion of K * P
    tokens) the architecture is identical to the parent. The final query-based
    readout is discarded; instead the current frame's contextualised tokens
    [B, P, D] = [B, 196, 256] are reshaped to [B, 256, 14, 14] and progressively
    upsampled (4 ConvTranspose2d strides, each 2x) to [B, n_anchors, 224, 224]
    logits.

    Decoding to 224x224 (cell_size ≈ 9.64 px in 2160 space) gives a denser
    readout than the 14x14 token grid (≈ 154 px per cell); the 3x3 windowed
    soft-argmax then yields sub-cell coordinates. The default
    ``heatmap_sigma_in_grid=16`` corresponds to ≈ 154 px (16 cells * 9.64 px).
    """
    def __init__(self, fusion_dim: int, n_anchors: int, grid: int = 224,
                 K: int = 3, n_temporal_layers: int = 2, n_heads: int = 4,
                 token_grid: int = 14):
        # Instantiate the parent with grid=token_grid (14) so the parent's
        # `time_queries` / `final_attn` aren't sized to 224 — but we never use
        # them in this subclass's forward.
        super().__init__(fusion_dim, n_anchors, grid=token_grid,
                         K=K, n_temporal_layers=n_temporal_layers,
                         n_heads=n_heads)
        # Output grid for the heatmap (not the token grid; the per-frame token
        # grid stays 14x14).
        self.grid = int(grid)             # 224 — used by callers for ce_loss
        self.token_grid = int(token_grid) # 14
        # Upsampler: 14->28->56->112->224, halving channels each step except
        # the last which projects to n_anchors logits.
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(fusion_dim, fusion_dim // 2,
                               kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(fusion_dim // 2, fusion_dim // 4,
                               kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(fusion_dim // 4, fusion_dim // 8,
                               kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(fusion_dim // 8, n_anchors,
                               kernel_size=4, stride=2, padding=1),
        )
        # Small init on the final layer so initial logits are near-uniform
        # (avoids saturated softmax at epoch 0 with 50K bins).
        nn.init.zeros_(self.upsampler[-1].bias)
        nn.init.normal_(self.upsampler[-1].weight, std=1e-3)

    def forward(self, vis_features_seq, text_features, text_mask=None):
        # Reuse the parent's prompt-conditioning + temporal fusion, then swap
        # in the upsampler instead of the query-based readout.
        if vis_features_seq.dim() == 3:
            vis_features_seq = vis_features_seq.unsqueeze(1)
            single_frame = True
        else:
            single_frame = False
        B, K, P, D = vis_features_seq.shape
        if not single_frame and K != self.K:
            raise ValueError(
                f"_PromptThenTemporalHead224 expects K={self.K} frames, "
                f"got K={K}")
        T = text_features.size(1)

        # 1) Per-frame prompt conditioning.
        vis_flat = vis_features_seq.reshape(B * K, P, D)
        text_expanded = text_features.unsqueeze(1).expand(B, K, T, D)\
                                     .reshape(B * K, T, D)
        if text_mask is not None:
            tm_bool = text_mask.bool()
            key_pad = ~tm_bool.unsqueeze(1).expand(B, K, T).reshape(B * K, T)
        else:
            key_pad = None
        q = self.norm_q_pc(vis_flat)
        kv = self.norm_kv_pc(text_expanded)
        attn_out, _ = self.vis_to_text_attn(q, kv, kv, key_padding_mask=key_pad)
        x = vis_flat + attn_out
        x = x + self.ffn_pc(self.norm_ffn_pc(x))
        x = x.reshape(B, K, P, D)

        # 2) Temporal fusion.
        if not single_frame:
            x = x + self.temporal_pos.view(1, K, 1, D)
            x = x.reshape(B, K * P, D)
            x = self.temporal_encoder(x)
            x = x.reshape(B, K, P, D)
        current = x[:, -1]                                                     # [B, P, D]

        # 3) 14->224 spatial upsample readout (replaces final_attn/einsum).
        side = self.token_grid
        # current: [B, P, D] -> [B, D, side, side]
        current_2d = current.transpose(1, 2).reshape(B, D, side, side)
        logits = self.upsampler(current_2d)                                    # [B, n_anchors, 224, 224]
        return logits


# ---------------------------------------------------------------------------
# Top-level model.
# ---------------------------------------------------------------------------

_SUPPORTED_ENCODERS = ("clip_clip",)
_SUPPORTED_HEADS = ("prompt_then_temporal_224",)
_UNSUPPORTED_MSG = (
    "only encoder_kind=clip_clip + head_kind=prompt_then_temporal_224 are "
    "supported"
)


class GazeTrajectoryPredictor(nn.Module):
    """The GazeMani gaze predictor.

    Encoder: frozen CLIP-B/16 (vision + text), followed by trainable
    per-modality fusion projections.

    Head: ``_PromptThenTemporalHead224`` (K=3 prompt-then-temporal head with
    14->224 spatial upsampler).
    """
    def __init__(self, encoder_kind="clip_clip",
                 fusion_dim=256, n_anchors=1, grid=224,
                 clip_model_id="openai/clip-vit-base-patch16",
                 head_kind="prompt_then_temporal_224",
                 temporal_K: int = 3,
                 n_temporal_layers: int = 2):
        super().__init__()
        if encoder_kind not in _SUPPORTED_ENCODERS:
            raise ValueError(
                f"unsupported encoder_kind={encoder_kind!r}: {_UNSUPPORTED_MSG}")
        # Only the prompt_then_temporal_224 head is supported.
        head_kind = head_kind or "prompt_then_temporal_224"
        if head_kind not in _SUPPORTED_HEADS:
            raise ValueError(
                f"unsupported head_kind={head_kind!r}: {_UNSUPPORTED_MSG}")

        self.encoder_kind = encoder_kind
        self.fusion_dim = fusion_dim
        self.n_anchors = n_anchors
        self.grid = grid
        self.head_kind = head_kind
        self.temporal_K = int(temporal_K)

        self.text_encoder = _ClipTextEncoder(clip_model_id, fusion_dim)
        self.vision_encoder = _ClipVisionEncoder(clip_model_id, fusion_dim)
        self.head = _PromptThenTemporalHead224(
            fusion_dim, n_anchors, grid=grid,
            K=self.temporal_K,
            n_temporal_layers=int(n_temporal_layers),
            token_grid=14)

    @property
    def is_temporal(self) -> bool:
        # The prompt-then-temporal head is always temporal.
        return True

    def encode_image(self, images):
        return self.vision_encoder(images)

    def encode_image_raw(self, images):
        return self.vision_encoder._raw(images)

    def encode_text(self, prompts):
        return self.text_encoder(prompts)

    def encode_text_raw(self, prompts):
        device = next(self.parameters()).device
        return self.text_encoder._raw(prompts, device)

    def forward(self, vis_features, text_features, text_mask=None):
        """``vis_features`` may be:
            * [B, P, D]       — single-frame fallback (head still applies prompt
                                cross-attn + readout, but skips temporal fusion).
            * [B, K, P, D]    — K-frame stack for the temporal head.

        ``D`` may be the raw vision-encoder dim (we then apply the trainable
        projection) or already ``fusion_dim``.

        Returns logits [B, n_anchors, G, G].
        """
        if vis_features.dim() == 4:
            B, K, P, D = vis_features.shape
            if D != self.fusion_dim:
                vis_features = self.vision_encoder.proj(
                    vis_features.reshape(B * K, P, D)).reshape(B, K, P, self.fusion_dim)
        else:
            if vis_features.size(-1) != self.fusion_dim:
                vis_features = self.vision_encoder.proj(vis_features)
        if text_features.size(-1) != self.fusion_dim:
            text_features = self.text_encoder.proj(text_features)
        return self.head(vis_features, text_features, text_mask=text_mask)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

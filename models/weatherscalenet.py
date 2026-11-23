"""WeatherScaleNet with shared station encoders and regional spatial exchange."""

import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp

from lib.utils import discretize_altitude, read_meta1, read_meta2


class WindowAttBlock(nn.Module):
    """Exchange station states within regions and across corresponding slots."""

    def __init__(self, hidden_size, num_heads, num, size, mlp_ratio=4.0, branch_mode="full", fusion_mode="learned"):
        super().__init__()
        self.branch_mode = branch_mode
        self.fusion_mode = fusion_mode
        self.num = num
        self.size = size
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.nnorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.snorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.weight_n = nn.Parameter(torch.tensor(0.0))
        self.weight_s = nn.Parameter(torch.tensor(0.0))
        self.concat_projection = nn.Linear(2 * hidden_size, hidden_size) if fusion_mode == "concat" else None

    def _run_local_branch(self, x):
        batch_size, steps, parts, patch_size, dims = x.shape
        qkv = self.snorm1(x.reshape(batch_size * steps * parts, patch_size, dims))
        x = x + self.sattn(qkv).reshape(batch_size, steps, parts, patch_size, dims)
        return x + self.smlp(self.snorm2(x))

    def _run_far_branch(self, x):
        batch_size, steps, parts, patch_size, dims = x.shape
        qkv = self.nnorm1(x.transpose(2, 3).reshape(batch_size * steps * patch_size, parts, dims))
        x = x + self.nattn(qkv).reshape(batch_size, steps, patch_size, parts, dims).transpose(2, 3)
        return x + self.nmlp(self.nnorm2(x))

    def fusion_weights(self):
        if self.fusion_mode == "fixed":
            return (0.5, 0.5)
        if self.fusion_mode == "learned":
            weights = torch.softmax(torch.stack([self.weight_n, self.weight_s]), dim=0)
            return (float(weights[0].detach().cpu()), float(weights[1].detach().cpu()))
        return None

    def forward(self, x):
        batch_size, steps, total_nodes, dims = x.shape
        if self.num * self.size != total_nodes:
            raise ValueError(f"Attention layout {self.num}x{self.size} does not match {total_nodes} patched nodes.")
        x = x.reshape(batch_size, steps, self.num, self.size, dims)

        if self.branch_mode == "local":
            x = self._run_local_branch(x)
        elif self.branch_mode == "far":
            x = self._run_far_branch(x)
        else:
            local_x = self._run_local_branch(x)
            far_x = self._run_far_branch(x)
            if self.fusion_mode == "concat":
                x = self.concat_projection(torch.cat([far_x, local_x], dim=-1))
            elif self.fusion_mode == "fixed":
                x = 0.5 * far_x + 0.5 * local_x
            elif self.fusion_mode == "learned":
                weights = torch.softmax(torch.stack([self.weight_n, self.weight_s]), dim=0)
                x = weights[0] * far_x + weights[1] * local_x
            else:
                raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        return x.reshape(batch_size, steps, total_nodes, dims)


class WeatherScaleNet(nn.Module):
    """Forecast each station from recent observations and geographic features."""

    def __init__(
        self,
        tem_patchsize,
        tem_patchnum,
        node_num,
        spa_patchsize,
        spa_patchnum,
        layers,
        factors,
        input_dims,
        node_dims,
        ori_parts_idx,
        reo_parts_idx,
        reo_all_idx,
        meta_file,
        ablation_mode="full",
        position_encoder="geoclip",
        use_elevation=True,
        elevation_encoder="discrete",
        branch_mode="full",
        fusion_mode="learned",
        input_channels=1,
        feature_seed=1,
        output_len=None,
    ):
        super().__init__()

        # Apply named ablations to the feature and spatial-exchange settings.
        if ablation_mode == "history-only":
            position_encoder, use_elevation, elevation_encoder = "none", False, "none"
        elif ablation_mode == "GeoCLIP-only":
            position_encoder, use_elevation, elevation_encoder = "geoclip", False, "none"
        elif ablation_mode == "elevation-only":
            position_encoder, use_elevation = "none", True
        elif ablation_mode == "local-only":
            branch_mode = "local"
        elif ablation_mode == "far-only":
            branch_mode = "far"

        self.ablation_mode = ablation_mode
        self.position_encoder_name = position_encoder
        self.use_elevation = use_elevation
        self.elevation_encoder_name = elevation_encoder if use_elevation else "none"
        self.branch_mode = branch_mode
        self.fusion_mode = fusion_mode
        self.tem_patchnum = tem_patchnum
        self.output_len = tem_patchsize if output_len is None else output_len
        self.node_num = node_num
        self.node_dims = node_dims
        self.input_channels = input_channels

        dims = input_dims + node_dims + node_dims
        self.embed_dims = dims
        self.input_st_fc = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_dims,
            kernel_size=(1, tem_patchsize),
            stride=(1, tem_patchsize),
            bias=True,
        )

        locations = torch.from_numpy(read_meta1(meta_file)).float().transpose(0, 1)
        self.register_buffer("locations", locations, persistent=False)
        self._build_position_encoder(position_encoder, node_dims, feature_seed)

        self.altitude_embedding = nn.Embedding(num_embeddings=50, embedding_dim=node_dims)
        self.altitude_mlp = (
            nn.Sequential(nn.Linear(1, node_dims), nn.GELU(), nn.Linear(node_dims, node_dims))
            if self.elevation_encoder_name == "raw-mlp"
            else None
        )
        altitude_indices, normalized_altitudes, has_altitude = self._build_altitude_features(meta_file, node_num)
        self.register_buffer("altitude_indices", altitude_indices, persistent=False)
        self.register_buffer("normalized_altitudes", normalized_altitudes, persistent=False)
        self.register_buffer("has_altitude", torch.tensor(has_altitude, dtype=torch.bool), persistent=False)

        if spa_patchnum % factors != 0:
            raise ValueError(f"spa_patchnum={spa_patchnum} must be divisible by factors={factors}.")
        self.spa_encoder = nn.ModuleList(
            [
                WindowAttBlock(
                    dims,
                    1,
                    spa_patchnum // factors,
                    spa_patchsize * factors,
                    mlp_ratio=1,
                    branch_mode=branch_mode,
                    fusion_mode=fusion_mode,
                )
                for _ in range(layers)
            ]
        )
        self.output_projection = nn.Linear(tem_patchnum * dims, self.output_len)

        self.register_buffer("ori_parts_idx", torch.as_tensor(ori_parts_idx, dtype=torch.long), persistent=False)
        self.register_buffer("reo_parts_idx", torch.as_tensor(reo_parts_idx, dtype=torch.long), persistent=False)
        self.register_buffer("reo_all_idx", torch.as_tensor(reo_all_idx, dtype=torch.long), persistent=False)

    def _build_position_encoder(self, name, node_dims, feature_seed):
        self.position_projection = None
        self.position_mlp = None
        self.station_embedding = None
        self.register_buffer("fixed_position_features", torch.empty(0), persistent=False)

        if name == "geoclip":
            from geoclip import LocationEncoder

            encoder = LocationEncoder().eval()
            with torch.no_grad():
                features = encoder(self.locations.cpu()).float()
            self.fixed_position_features = features
            self.position_projection = nn.Linear(features.shape[-1], node_dims)
        elif name == "random-geoclip":
            generator = torch.Generator().manual_seed(feature_seed)
            self.fixed_position_features = torch.randn(self.node_num, 512, generator=generator)
            self.position_projection = nn.Linear(512, node_dims)
        elif name == "raw-mlp":
            self.position_mlp = nn.Sequential(nn.Linear(2, node_dims), nn.GELU(), nn.Linear(node_dims, node_dims))
        elif name == "sincos":
            self.position_projection = nn.Linear(4, node_dims)
        elif name == "rff":
            generator = torch.Generator().manual_seed(feature_seed)
            frequencies = torch.randn(2, node_dims, generator=generator) * 4.0
            phases = torch.rand(node_dims, generator=generator) * (2.0 * math.pi)
            self.register_buffer("rff_frequencies", frequencies, persistent=False)
            self.register_buffer("rff_phases", phases, persistent=False)
            self.position_projection = nn.Linear(node_dims, node_dims)
        elif name == "station-id":
            self.station_embedding = nn.Embedding(self.node_num, node_dims)
        elif name != "none":
            raise ValueError(f"Unknown position encoder: {name}")

    def _normalized_locations(self):
        scale = self.locations.new_tensor([90.0, 180.0])
        return self.locations / scale

    def _position_features(self):
        name = self.position_encoder_name
        if name in {"geoclip", "random-geoclip"}:
            return self.position_projection(self.fixed_position_features)
        if name == "raw-mlp":
            return self.position_mlp(self._normalized_locations())
        if name == "sincos":
            radians = torch.deg2rad(self.locations)
            features = torch.stack(
                [torch.sin(radians[:, 0]), torch.cos(radians[:, 0]), torch.sin(radians[:, 1]), torch.cos(radians[:, 1])],
                dim=-1,
            )
            return self.position_projection(features)
        if name == "rff":
            features = math.sqrt(2.0 / self.node_dims) * torch.cos(
                self._normalized_locations() @ self.rff_frequencies + self.rff_phases
            )
            return self.position_projection(features)
        if name == "station-id":
            return self.station_embedding.weight
        return self.locations.new_zeros((self.node_num, self.node_dims))

    @staticmethod
    def _build_altitude_features(meta_file, node_num):
        altitudes = read_meta2(meta_file)
        if altitudes is None:
            return torch.zeros(node_num, dtype=torch.long), torch.zeros(node_num, 1), False
        altitudes = np.asarray(altitudes, dtype=np.float32)
        finite = np.isfinite(altitudes)
        fill = float(np.nanmedian(altitudes)) if np.any(finite) else 0.0
        altitudes = np.where(finite, altitudes, fill)
        normalized = (altitudes - np.mean(altitudes)) / max(float(np.std(altitudes)), 1e-6)
        return (
            torch.as_tensor(discretize_altitude(altitudes, num_bins=50), dtype=torch.long),
            torch.as_tensor(normalized[:, None], dtype=torch.float32),
            True,
        )

    def _expand_node_features(self, batch_size, steps):
        position_emb = self._position_features()
        if self.elevation_encoder_name == "discrete" and bool(self.has_altitude):
            altitude_emb = self.altitude_embedding(self.altitude_indices)
        elif self.elevation_encoder_name == "raw-mlp" and bool(self.has_altitude):
            altitude_emb = self.altitude_mlp(self.normalized_altitudes)
        elif self.elevation_encoder_name not in {"none", "discrete", "raw-mlp"}:
            raise ValueError(f"Unknown elevation encoder: {self.elevation_encoder_name}")
        else:
            altitude_emb = position_emb.new_zeros((self.node_num, self.node_dims))
        position_emb = position_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, steps, -1, -1)
        altitude_emb = altitude_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, steps, -1, -1)
        return position_emb, altitude_emb

    def embedding(self, x):
        batch_size = x.shape[0]
        history = self.input_st_fc(x.transpose(1, 3)).transpose(1, 3)
        steps = history.shape[1]
        position_emb, altitude_emb = self._expand_node_features(batch_size, steps)
        return torch.cat([history, position_emb, altitude_emb], dim=-1)

    def get_fusion_weights(self):
        return [block.fusion_weights() for block in self.spa_encoder]

    def forward(self, x):
        embedded_x = self.embedding(x)
        patched_x = embedded_x[:, :, self.reo_all_idx, :]
        for block in self.spa_encoder:
            patched_x = block(patched_x)

        restored_x = embedded_x.new_zeros(
            embedded_x.shape[0], embedded_x.shape[1], self.node_num, self.embed_dims
        )
        # Match the destination dtype when restoring station order under AMP.
        restored_x[:, :, self.ori_parts_idx, :] = patched_x[:, :, self.reo_parts_idx, :].to(
            restored_x.dtype
        )
        restored_x = restored_x.permute(0, 2, 1, 3).reshape(embedded_x.shape[0], self.node_num, -1)
        return self.output_projection(restored_x).permute(0, 2, 1).unsqueeze(-1)

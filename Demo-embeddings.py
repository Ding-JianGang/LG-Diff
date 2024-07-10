from typing import Type, Tuple, Optional

import numpy as np
from torch import nn
import torch
from torch.nn import functional as F

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

# class LocationEmbedding(nn.Module):

#     def __init__(
#         self, loc_num: int = 513, loc_embed_dim: int = 2048,
#     ):
#         super().__init__()
        
#         self.x_axis_embedding = nn.Embedding(1, loc_embed_dim // 2)
#         self.y_axis_embedding = nn.Embedding(1, loc_embed_dim // 2)
#         self.location_token = nn.Embedding(loc_num, loc_embed_dim // 2)

#         self.loc_mlp = MLPBlock(embedding_dim=loc_embed_dim, mlp_dim=loc_embed_dim)

#     def forward(self, point):
#         x_embed = self.location_token(point[0]) + self.x_axis_embedding.weight
#         y_embed = self.location_token(point[1]) + self.y_axis_embedding.weight
        
#         fusion_xy = torch.cat([x_embed, y_embed], dim=-1)
#         out = self.loc_mlp(fusion_xy)
#         return out

class LocationEmbedding(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 1024, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )
        self.point_embedding = nn.Embedding(1, num_pos_feats * 2)
        self.extra_mlp = MLP(num_pos_feats * 2, num_pos_feats, num_pos_feats * 2, 3)

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_test(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.flatten(0, 1)  # HW, C

    def forward(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]=[512, 512], without_mlp=False,
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone().to(torch.float)
        coords[:, 0] = coords[:, 0] / image_size[1]
        coords[:, 1] = coords[:, 1] / image_size[0]

        if without_mlp:
            return self._pe_encoding(coords.to(torch.float))

        out = self.extra_mlp(self._pe_encoding(coords.to(torch.float)) + self.point_embedding.weight)
        return out
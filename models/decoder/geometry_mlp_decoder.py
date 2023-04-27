import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..networks import init_seq, positional_encoding

from gridencoder import GridEncoder


class GeometryMlpDecoder(nn.Module):
    VALID_FEATURES = {"density", "normal", "frame", "uv", "uv_weights", "brdf"}

    def __init__(
        self,
        code_dim,
        pos_freqs,
        uv_dim,
        uv_count,
        brdf_dim,
        hidden_size,
        num_layers,
        requested_features={"density", "normal"},  # density, normal/frame, uv, brdf
        use_bias=False
    ):
        super().__init__()

        assert code_dim >= 0
        assert uv_dim >= 0
        if uv_dim > 0:
            assert uv_dim >= 1
        if uv_count > 1:
            assert "uv_weights" in requested_features
        if "uv_weights" in requested_features:
            assert uv_count > 1
        assert hidden_size >= 1
        assert num_layers >= 1
        # diffuse(3)
        # diffuse(3), roughness(1), specular(1)
        # diffuse(3), roughness(1), specular(3)
        # diffuse(3), roughness(1), metallic(1), specular(1)
        assert brdf_dim in [3, 5, 6, 7]

        assert not ("normal" in requested_features and "frame" in requested_features)

        self.code_dim = code_dim
        self.uv_dim = uv_dim
        self.uv_count = uv_count
        self.requested_features = requested_features
        self.input_channels = code_dim + 32
        self.brdf_dim = brdf_dim

        self.output_dim = 0
        if "density" in requested_features:
            self.output_dim += 1
        if "uv" in requested_features:
            self.output_dim += self.uv_dim * self.uv_count
        if "normal" in requested_features:
            self.output_dim += 3
        if "frame" in requested_features:
            self.output_dim += 4
        if "brdf" in requested_features:
            self.output_dim += self.brdf_dim

        block = []
        block.append(GridEncoder(input_dim=3,
                                    num_levels=16, level_dim=2,
                                    log2_hashmap_size=19,
                                    base_resolution=16, desired_resolution=2048))
        block.append(nn.Linear(self.input_channels, hidden_size, bias=use_bias))
        block.append(nn.ReLU())
        for i in range(num_layers):
            block.append(nn.Linear(hidden_size, hidden_size, bias=use_bias))
            block.append(nn.ReLU())
        block.append(nn.Linear(hidden_size, self.output_dim, bias=use_bias))
        self.block = nn.Sequential(*block)
        init_seq(self.block)

    def forward(self, input_code, pts, require_grad=False):
        """
        Args:
            input_code: :math:`(N,E)`
            pts: :math:`(N,Rays,Samples,3)`
        """
        assert input_code is None or len(input_code.shape) == 2
        assert len(pts.shape) == 4
        assert pts.shape[-1] == 3
        assert input_code is None or input_code.shape[0] == pts.shape[0]

        if require_grad:
            with torch.enable_grad():
                pts = pts.clone()
                pts.requires_grad_(True)
                self.output = self.block(pts)
        else:
            self.output = self.block(pts)

        output = {}
        index = 0
        if "density" in self.requested_features:
            if require_grad:
                with torch.enable_grad():
                    sigma = self.output[..., 0]
                    unused = torch.empty_like(sigma, requires_grad=False)
                    grad = torch.autograd.grad(sigma, pts, unused, retain_graph=True)[0]
                    grad = -F.normalize(grad, dim=-1, eps=1e-6)
                    output["sigma_grad"] = grad.detach().clone()
            else:
                sigma = self.output[..., 0]

            output["density"] = F.softplus(sigma)
            index += 1
        if "uv" in self.requested_features:
            output["uv"] = torch.tanh(
                self.output[..., index : index + self.uv_dim * self.uv_count]
            )
            index += self.uv_dim * self.uv_count

        if "uv_weights" in self.requested_features:
            output["uv_weights_logits"] = self.output[
                ..., index : index + self.uv_count
            ]
            output["uv_weights"] = F.softmax(output["uv_weights_logits"], dim=-1)
            index += self.uv_count
        if "normal" in self.requested_features:
            output["normal"] = F.normalize(
                torch.tanh(self.output[..., index : index + 3]), dim=-1
            )
            index += 3
        if "frame" in self.requested_features:
            output["frame"] = F.normalize(self.output[..., index : index + 4], dim=-1)
            index += 4
        if "brdf" in self.requested_features:
            output["brdf"] = torch.sigmoid(
                self.output[..., index : index + self.brdf_dim]
            )
            index += self.brdf_dim

        assert index == self.output_dim

        return output

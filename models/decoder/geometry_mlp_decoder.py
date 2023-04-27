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
        pred_normal=False,
        use_bias=False
    ):
        super().__init__()

        # self.code_dim = code_dim
        # self.uv_dim = uv_dim
        # self.uv_count = uv_count
        # self.requested_features = requested_features
        # self.input_channels = code_dim + 32
        # self.brdf_dim = brdf_dim

        # self.output_dim = 
        # if "density" in requested_features:
        #     self.output_dim += 1
        # if "uv" in requested_features:
        #     self.output_dim += self.uv_dim * self.uv_count
        # if "normal" in requested_features:
        #     self.output_dim += 3
        # if "frame" in requested_features:
        #     self.output_dim += 4
        # if "brdf" in requested_features:
        #     self.output_dim += self.brdf_dim

        self.pos_encoder = GridEncoder(input_dim=3,
                                    num_levels=16, level_dim=2,
                                    log2_hashmap_size=19,
                                    base_resolution=16, desired_resolution=2048)
        block = []
        block.append(nn.Linear(32, hidden_size, bias=use_bias))
        block.append(nn.ReLU())
        for i in range(num_layers):
            block.append(nn.Linear(hidden_size, hidden_size, bias=use_bias))
            block.append(nn.ReLU())
        block.append(nn.Linear(hidden_size, 1, bias=use_bias))
        self.sigma_block = nn.Sequential(*block)
        init_seq(self.sigma_block)

        self.normal_block = None
        if pred_normal:
            block3 = []
            block3.append(nn.Linear(32, hidden_size, bias=use_bias))
            block3.append(nn.ReLU())
            for i in range(2):
                block3.append(nn.Linear(hidden_size, hidden_size, bias=use_bias))
                block3.append(nn.ReLU())
            block3.append(nn.Linear(hidden_size, 3, bias=use_bias))
            self.normal_block = nn.Sequential(*block3)
            init_seq(self.normal_block)

    def forward(self, input_code, pts, require_grad=False):
        """
        Args:
            input_code: :math:`(N,E)`
            pts: :math:`(N,Rays,Samples,3)`
        """

        if require_grad:
            with torch.enable_grad():
                pts = pts.clone()
                pts.requires_grad_(True)
                h = self.pos_encoder(pts)
                sigma = self.sigma_block(h)
        else:
            h = self.pos_encoder(pts)
            sigma = self.sigma_block(h)

        output = {}
        
        if require_grad:
            with torch.enable_grad():
                unused = torch.empty_like(sigma, requires_grad=False)
                grad = torch.autograd.grad(sigma, pts, unused, retain_graph=True)[0]
                grad = -F.normalize(grad, dim=-1, eps=1e-6)
                output["sigma_grad"] = grad.detach().clone()

        output["density"] = F.softplus(sigma)

        # if "uv" in self.requested_features:
        #     output["uv"] = torch.tanh(
        #         self.output[..., index : index + self.uv_dim * self.uv_count]
        #     )
        #     index += self.uv_dim * self.uv_count

        # if "uv_weights" in self.requested_features:
        #     output["uv_weights_logits"] = self.output[
        #         ..., index : index + self.uv_count
        #     ]
        #     output["uv_weights"] = F.softmax(output["uv_weights_logits"], dim=-1)
        #     index += self.uv_count
        if self.normal_block is not None:
            normal = self.normal_block(h)
            output["normal"] = F.normalize(normal, dim=-1, eps=1e-6)
            # index += 3
        # if "frame" in self.requested_features:
        #     output["frame"] = F.normalize(self.output[..., index : index + 4], dim=-1)
        #     index += 4
        # if "brdf" in self.requested_features:
        #     output["brdf"] = torch.sigmoid(
        #         self.output[..., index : index + self.brdf_dim]
        #     )
        #     index += self.brdf_dim

        # assert index == self.output_dim

        return output

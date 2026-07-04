import logging
import torch
from math import pi


class Embedder:
    """
    borrow from
    https://github.com/zju3dv/animatable_nerf/blob/master/lib/networks/embedder.py
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(input_dims, num_freqs, include_input=True, log_sampling=True):
    embed_kwargs = {
        "input_dims": input_dims,
        "num_freqs": num_freqs,
        "max_freq_log2": num_freqs - 1,
        "include_input": include_input,
        "log_sampling": log_sampling,
        "periodic_fns": [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    logging.debug(f"embedder out dim = {embedder_obj.out_dim}")
    return embedder_obj


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device, dtype=x.dtype)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class SpatialPoseEmbedder(torch.nn.Module):
    """
    Embeds a relative pose matrix into a spatial feature map.

    This module takes a 4x4 relative pose matrix and generates a 2D feature map
    that can be added to other spatial features (like an image feature map).
    The process involves:
    1. Creating a normalized coordinate grid.
    2. Transforming this grid using the input pose matrix.
    3. Encoding the transformed coordinates into high-frequency features using
       Gaussian Fourier Feature mapping.
    4. Projecting these features to the desired number of output channels.
    """
    def __init__(self, output_channels, fourier_mapping_size=128):
        super().__init__()
        # Input to the Fourier transform will be 2D coordinates (x, y), so input_channels=2.
        # The output of the Fourier transform has 2x the mapping size due to sin and cos pairs.
        fourier_channels = 2 * fourier_mapping_size
        
        self.fourier_transform = GaussianFourierFeatureTransform(
            num_input_channels=2, mapping_size=fourier_mapping_size
        )
        
        # A 1x1 convolution to project the Fourier features to the final output channel dimension.
        self.projection = torch.nn.Conv2d(fourier_channels, output_channels, kernel_size=1)

    def forward(self, rel_pose, height, width, device):
        # rel_pose shape: (B, 4, 4)
        
        # 1. Create a coordinate grid.
        # Create a normalized grid of coordinates, from -1 to 1.
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing='ij'
        ) # shape: (H, W)
        
        # Stack to create 3D homogeneous coordinates (x, y, 0, 1).
        # shape: (H, W, 4)
        coords_3d = torch.stack([
            x_coords, y_coords, torch.zeros_like(x_coords), torch.ones_like(x_coords)
        ], dim=-1)
        
        # Reshape for batch matrix multiplication: (H, W, 4) -> (1, H, W, 4, 1)
        coords_3d = coords_3d.unsqueeze(0).unsqueeze(-1)
        
        # 2. Transform coordinates using the pose matrix.
        # Reshape pose matrix for broadcasting: (B, 4, 4) -> (B, 1, 1, 4, 4)
        b, _, _ = rel_pose.shape
        pose_matrix = rel_pose.view(b, 1, 1, 4, 4)
        
        # Apply the transformation via matrix multiplication.
        # Broadcasting handles the batch dimension.
        # (B, 1, 1, 4, 4) @ (1, H, W, 4, 1) -> (B, H, W, 4, 1)
        transformed_coords_3d = pose_matrix @ coords_3d.to(pose_matrix.dtype)
        transformed_coords_3d = transformed_coords_3d.squeeze(-1) # shape: (B, H, W, 4)
        
        # We only need the transformed x and y coordinates for the 2D feature map.
        # shape: (B, H, W, 2)
        transformed_coords_2d = transformed_coords_3d[..., :2]
        
        # Permute to (B, C, H, W) format for convolution layers.
        # (B, H, W, 2) -> (B, 2, H, W)
        transformed_coords_2d = transformed_coords_2d.permute(0, 3, 1, 2)
        
        # 3. Encode coordinates with Fourier features.
        # (B, 2, H, W) -> (B, fourier_channels, H, W)
        pose_features = self.fourier_transform(transformed_coords_2d)
        
        # 4. Project features to the final channel size.
        # (B, fourier_channels, H, W) -> (B, output_channels, H, W)
        pose_condition_map = self.projection(pose_features)
        
        return pose_condition_map
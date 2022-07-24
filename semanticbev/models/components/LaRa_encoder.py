import torch
from torch import nn
from einops import rearrange, repeat
from typing import Tuple, Dict

from semanticbev.models.components.layers import Up, Sequential, cross_attention_layer, self_attention_block

class InputEmbedding(nn.Module):
    def __init__(self, num_input_channels):
        super().__init__()
        self._num_input_channels = num_input_channels

    @property
    def num_input_channels(self):
        return self._num_input_channels

    def forward(self, x, **kwargs):
        raise NotImplementedError()

class InputEmbeddingCollection(InputEmbedding):
    def __init__(self, num_image_channels: int, input_embeddings: Dict[str, InputEmbedding], embedding_map: str = 'mlp',
                 embedding_channels: int = 128, embedding_merge: str = 'concat', input_prenorm = False):

        input_embeddings = input_embeddings.values()
        raw_embedding_channels = sum([a.num_input_channels for a in input_embeddings])
        if embedding_map == 'identity':
            embedding_channels = raw_embedding_channels

        if embedding_merge == 'concat':
            num_input_channels = num_image_channels + embedding_channels
        elif embedding_merge == 'add':
            if embedding_map == 'identity':
                assert raw_embedding_channels == num_image_channels

            print("`embedding_merge` is 'add' -> setting `embedding_channels` to `num_image_channels` ")
            embedding_channels = num_image_channels
            num_input_channels = num_image_channels
        else:
            raise ValueError("embedding_merge not in ['concat', 'add']")
        self.embedding_merge = embedding_merge

        super().__init__(num_input_channels=num_input_channels)

        if input_prenorm:
            self.prenorm = nn.LayerNorm(num_image_channels)
        else:
            self.prenorm = nn.Identity()

        if embedding_map == 'identity':
            self.embedding_map = nn.Identity()
        elif embedding_map == 'linear':
            self.embedding_map = nn.Linear(raw_embedding_channels, embedding_channels)
        elif embedding_map == 'mlp':
            self.embedding_map = Sequential(
                nn.Linear(raw_embedding_channels, embedding_channels),
                nn.GELU(),
                nn.Linear(embedding_channels, embedding_channels)
            )
        else:
            raise ValueError(f"embedding_map not in ['identity', 'linear', 'mlp']: {embedding_map}")

        self.input_embeddings = nn.ModuleList(input_embeddings)

    def forward(self, x, **kwargs):

        b, n, *_ = x.shape

        embeddings = []
        for embed in self.input_embeddings:
            embedding = embed(x, **kwargs)
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=-1)

        embeddings = self.embedding_map(embeddings)

        x = rearrange(x, 'b ... c -> b (...) c')
        x = self.prenorm(x)

        if self.embedding_merge == 'concat':
            return torch.cat([x, embeddings], dim=-1)
        elif self.embedding_merge == 'add':
            return x + embeddings
        else:
            raise ValueError(f"embedding_merge not in ['concat', 'add']: {self.embedding_merge}")


class LaRaEncoder(nn.Module):
    def __init__(
            self,
            input_embedding: InputEmbedding,
            latent_shape: Tuple[int, int],
            num_layers: int,
            num_cross_attention_heads: int = 4,
            num_self_attention_heads: int = 4,
            num_self_attention_layers_per_block: int = 2,
            dropout: float = 0.0,
            activation_checkpoint: bool = False,
    ):
        """
        Args:
            input_embedding: Add or concatenate an embedding to the input.
                             Also transforms the input images to a sequence of vectors (B, N, H, W, C) -> (B, N*H*W, C)
                             with B the batch size, N the number of cameras, H images height, W images width and C the
                             number of input channels.
            latent_shape: Shape of the latent array, (K, D), where N is the number of latent variables
                          and D the number of latent channels.
            num_layers: Number of self-attention block.
            num_cross_attention_heads: Number of cross-attention heads.
            num_self_attention_heads: Number of self-attention heads.
            num_self_attention_layers_per_block: Number of self-attention layers per self-attention block.
            dropout: Dropout for self- and cross-attention layers and residuals.
            activation_checkpoint: If True, implements an activation checkpoint for each self-attention layer
                                   and cross-attention layer (doesn't works well with DDP).
        """
        super().__init__()

        self.input_embedding = input_embedding
        self.num_layers = num_layers

        num_latent_channels = latent_shape[1]


        self.layer_1 = Sequential(
            cross_attention_layer(
                num_q_channels=num_latent_channels,
                num_kv_channels=input_embedding.num_input_channels,
                num_heads=num_cross_attention_heads,
                dropout=dropout,
                activation_checkpoint=activation_checkpoint,
            ),
            self_attention_block(
                num_layers=num_self_attention_layers_per_block,
                num_channels=num_latent_channels,
                num_heads=num_self_attention_heads,
                dropout=dropout,
                activation_checkpoint=activation_checkpoint,
            )
        )

        if num_layers > 1:
            # will be used recurrently depending on num_layers
            self.layer_n = self_attention_block(
                    num_layers=num_self_attention_layers_per_block,
                    num_channels=num_latent_channels,
                    num_heads=num_self_attention_heads,
                    dropout=dropout,
                    activation_checkpoint=activation_checkpoint,
            )


        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(*latent_shape))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, rots, trans, intrins, input_stride, pad_mask=None):
        b, *_ = x.shape

        # encode task-specific input
        x = self.input_embedding(x, rots=rots, trans=trans, intrins=intrins, input_stride=input_stride)

        # repeat initial latent vector along batch dimension
        x_latent = repeat(self.latent, '... -> b ...', b=b)

        x_latent = self.layer_1(x_latent, x, pad_mask)
        for _ in range(self.num_layers - 1):
            x_latent = self.layer_n(x_latent, pad_mask)


        return x_latent



import torch
from torch import nn
from typing import Tuple, Dict

from semanticbev.models.components.layers import cross_attention_layer, Sequential



class QueryGenerator(nn.Module):
    def __init__(self, query_seq_len, num_input_channels):
        super().__init__()
        self._query_seq_len = query_seq_len
        self._num_input_channels = num_input_channels

    @property
    def query_seq_len(self):
        return self._query_seq_len

    @property
    def num_input_channels(self):
        return self._num_input_channels

    def forward(self, batch_size, device, **kwargs):
        raise NotImplementedError()

class QueryGeneratorCollection(QueryGenerator):
    def __init__(self, query_generators: Dict[str, QueryGenerator]):

        query_generators = query_generators.values()
        num_input_channels = sum([q.num_input_channels for q in query_generators])
        query_seq_lens = [q.query_seq_len for q in query_generators]
        if len(set(query_seq_lens)) > 1:
            raise ValueError("All QueryGenerator in a QueryGeneratorCollection must"
                             " have the same value for query_seq_len.")

        super().__init__(query_seq_len=query_seq_lens[0], num_input_channels=num_input_channels)

        self.query_generators = nn.ModuleList(query_generators)

    def forward(self, batch_size, device, **kwargs):

        queries = []
        for generator in self.query_generators:
            query = generator(batch_size, device, **kwargs)
            queries.append(query)

        return torch.cat(queries, dim=-1)

class OutputAdapter(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self._output_shape = output_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        raise NotImplementedError()


class LaRaDecoder(nn.Module):
    def __init__(
            self,
            output_adapter: OutputAdapter,
            query_generator: QueryGenerator,
            latent_shape: Tuple[int, int],  # as produced by model encoder
            num_cross_attention_heads: int = 4,
            dropout: float = 0.0,
            activation_checkpoint: bool = False,
            query_map: str = 'mlp',
            residual_ca: bool = True,
    ):
        """
        output_adapter: Transforms the decoder output of shape (B, M, C) to output shape.
        query_generator: Defines the embedding to use as query for the decoder cross-attention
        latent_shape: Shape of the latent array (K, D)
        num_cross_attention_heads: Number of cross-attention heads.
        dropout: Dropout for cross-attention layers and residuals.
        activation_checkpoint: If True, implements an activation checkpoint for the decoder's cross-attention layer.
        query_map: Operation to apply on the query embedding before cross-attention 'identity', 'linear' or 'mlp'
        residual_ca: Whether the cross-attention of the decoder is residual or not. If the query is a set of coordinates
                     it may not make sense to integrate it back into the features.
        """
        super().__init__()

        num_latent_channels = latent_shape[-1]
        num_output_channels = output_adapter.output_shape[-1]

        self.output_adapter = output_adapter
        self.latent_shape = latent_shape
        self.cross_attention = cross_attention_layer(
            num_q_channels=num_output_channels,
            num_kv_channels=num_latent_channels,
            num_heads=num_cross_attention_heads,
            dropout=dropout,
            activation_checkpoint=activation_checkpoint,
            residual_ca=residual_ca,
        )

        self.query_generator = query_generator

        if query_map == 'identity':
            self.query_map = nn.Identity()
        elif query_map == 'linear':
            self.query_map = nn.Linear(self.query_generator.num_input_channels, num_output_channels)
        elif query_map == 'mlp':
            self.query_map = Sequential(
                nn.Linear(self.query_generator.num_input_channels, num_output_channels),
                nn.GELU(),
                nn.Linear(num_output_channels, num_output_channels)
            )
        else:
            raise ValueError(f"query_map not in ['identity', 'linear', 'mlp']: {query_map}")

    def forward(self, latent_array, **kwargs):
        b, *d = latent_array.shape

        if tuple(d) != tuple(self.latent_shape):
            raise ValueError(f'Latent shape {d} different from required shape {self.latent_shape}')

        query = self.query_generator(b, latent_array.device, **kwargs)

        query = self.query_map(query)

        output = self.cross_attention(query, latent_array)

        return self.output_adapter(output)

"""
The main difference between `RotateChord` and `RotateChordVarLen` is how they handle sequences in a batch:

1. `RotateChord` operates on fixed-length sequences in a batch.
    It rolls all sequences in the batch jointly along the time dimension.
    This means that the same roll operation (with the same offsets) is applied to all sequences in the batch.

2. `RotateChordVarLen` operates on variable-length sequences in a batch.
    It rolls each sequence separately along the time dimension, considering its specific length.
    This is useful when you have a batch containing sequences of extremely different lengths,
    and you want to apply the roll operation independently to each sequence according to its length.

In summary, `RotateChord` is designed for batches with fixed-length sequences, while `RotateChordVarLen` is designed for batches with variable-length sequences.
"""

import math
import torch
import numpy as np
from torch import nn

class Mlp(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) module in PyTorch.

    Args:
    in_features (int): Number of input features.
    hidden_features (int, optional): Hidden layer size. Defaults to in_features.
    out_features (int, optional): Output size. Defaults to in_features.
    act_layer (nn.Module, optional): Activation function. Defaults to nn.GELU.
    drop (float, optional): Dropout probability. Defaults to 0.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
   
    
class BatchNorm(nn.Module):
    """
    A PyTorch module implementing 1D Batch Normalization for token embeddings.

    Args:
        embedding_size (int): The size of the token embeddings.
    """
    def __init__(self, embedding_size):
        super().__init__()
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.bn(x)
        x = torch.permute(x, (0, 2, 1))
        return x


class GroupNorm(nn.Module):
    """
    A PyTorch module implementing Group Normalization for token embeddings.

    Args:
        embedding_size (int): The size of the token embeddings.
        n_groups (int): The number of groups to divide the channels into.
    """
    def __init__(self, embedding_size, n_groups):
        super().__init__()
        self.gn = nn.GroupNorm(n_groups, embedding_size)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.gn(x)
        x = torch.permute(x, (0, 2, 1))
        return x


def map_norm(norm_type, embedding_size, track_size=None):
    """
    Maps the given normalization type to the corresponding PyTorch module.

    Args:
        norm_type (str): The normalization type ('LN', 'BN', 'GN', or None).
        embedding_size (int): The size of the token embeddings.
        track_size (int, optional): The number of groups for Group Normalization.

    Returns:
        nn.Module: The corresponding normalization module.
    """
    if norm_type == 'LN':
        norm = nn.LayerNorm(embedding_size)
    elif norm_type == 'BN':
        norm = BatchNorm(embedding_size)
    elif norm_type == 'GN':
        norm = GroupNorm(embedding_size, track_size)
    elif norm_type == 'None':
        norm = nn.Identity()
    return norm


class RotateChord(nn.Module):
    """
    A PyTorch module that performs a parameter-free rotation of tracks within token embeddings.

    This module can be used to augment or modify the input data in a data-driven manner. The rotation is
    performed jointly for all sequences in a batch and is based on powers of 2 (Chord protocol).

    Args:
        track_size (int): The size of tracks to be rotated.
    """
    def __init__(self, track_size):
        super().__init__()
        self.track_size = track_size

    def forward(self, x, lengths=None):
        y = torch.split(
            tensor=x,
            split_size_or_sections=self.track_size,
            dim=-1
        )

        # Roll sequences in a batch jointly
        # The first track remains unchanged
        z = [y[0]]
        for i in range(1, len(y)):
            offset = - 2 ** (i - 1)
            z.append(torch.roll(y[i], shifts=offset, dims=1))

        z = torch.cat(z, -1)
        return z

    
class RotateChordVarLen(nn.Module):
    """
    A PyTorch module that performs a parameter-free rotation of tracks within variable-length token embeddings.

    This module can be used to augment or modify the input data in a data-driven manner. The rotation is
    performed separately for all sequences in a batch and is based on powers of 2. This version is designed to
    handle variable-length input sequences of extremely diverse range.
    
    No padding is applied.

    Args:
        track_size (int): The size of tracks to be rotated.
    """
    def __init__(self, track_size):
        super().__init__()
        self.track_size = track_size

    def forward(self, x, lengths):
        ys = torch.split(
            tensor=x,
            split_size_or_sections=lengths.tolist(),
            dim=0
        )

        zs = []

        # Roll sequences separately
        for y in ys:
            y = torch.split(
                tensor=y,
                split_size_or_sections=self.track_size,
                dim=-1
            )
            z = [y[0]]
            for i in range(1, len(y)):
                offset = -2 ** (i - 1)
                z.append(torch.roll(y[i], shifts=offset, dims=0))
            z = torch.cat(z, -1)
            zs.append(z)

        z = torch.cat(zs, 0)
        return z
    

class ChordMixerBlock(nn.Module):
    """
    A PyTorch module implementing the ChordMixerBlock.

    This module combines two main steps in the ChordMixer layer: Rotate and Mix.
    The dropout between too is added.

    Args:
        embedding_size (int): The size of the token embeddings.
        track_size (int): The size of tracks to be rotated.
        hidden_size (int): The hidden layer size for the MLP.
        mlp_dropout (float): The dropout probability for the MLP.
        layer_dropout (float): The dropout probability for the ChordMixerBlock.
        prenorm (str): The type of normalization for the pre-normalization step.
        norm (str): The type of normalization for the post-normalization step.
        var_len (bool): Whether to use variable-length input sequences.
    """

    def __init__(
        self,
        embedding_size,
        track_size,
        hidden_size,
        mlp_dropout,
        layer_dropout,
        prenorm,
        norm,
        var_len=False
    ):
        super().__init__()
        self.prenorm = map_norm(prenorm, embedding_size, track_size)
        self.norm = map_norm(norm, embedding_size, track_size)
         
        self.mix = Mlp(
            embedding_size,
            hidden_size,
            embedding_size,
            act_layer=nn.GELU,
            drop=mlp_dropout
        )

        self.dropout = nn.Dropout(layer_dropout)
        if var_len:
            self.rotate = RotateChordVarLen(track_size)
        else:
            self.rotate = RotateChord(track_size)
    
    def forward(self, x, lengths=None):
        res_con = x
        x = self.prenorm(x)
        x = self.mix(x)
        x = self.dropout(x)
        x = self.rotate(x, lengths)
        x = x + res_con
        x = self.norm(x)
        return x
    

class ChordMixerEncoder(nn.Module):
    """
    A PyTorch module implementing a ChordMixer Encoder as a stack of ChordMixer layers. 
    The number of layers in the stack is determined by the maximum sequence length in the batch.
    The number of layers is fixed for the equal lengths mode.

    Args:
        max_seq_len (int): The maximum sequence length of the input tensor.
        track_size (int): The size of tracks to be rotated.
        hidden_size (int): The hidden layer size for the MLP.
        mlp_dropout (float): The dropout probability for the MLP.
        layer_dropout (float): The dropout probability for the ChordMixerBlock.
        prenorm (str): The type of normalization for the pre-normalization step.
        norm (str): The type of normalization for the post-normalization step.
        var_len (bool): Whether to use variable-length input sequences.
    """
    def __init__(
        self,
        max_seq_len,
        track_size,
        hidden_size,
        mlp_dropout,
        layer_dropout,
        prenorm,
        norm,
        var_len=False
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_n_layers = math.ceil(np.log2(max_seq_len))
        embedding_size = int((self.max_n_layers + 1) * track_size)
        self.var_len = var_len
        self.chordmixer_blocks = nn.ModuleList(
            [
                ChordMixerBlock(
                    embedding_size,
                    track_size,
                    hidden_size,
                    mlp_dropout,
                    layer_dropout,
                    prenorm,
                    norm,
                    var_len
                )
                for _ in range(self.max_n_layers)
            ]
        )

    def forward(self, x, lengths=None):
        # If var_len, use a variable number of layers
        if self.var_len:
            n_layers = torch.ceil(torch.log2(lengths[0])).detach().cpu().int()
        else:
            n_layers = self.max_n_layers

        for layer in range(n_layers):
            x = self.chordmixer_blocks[layer](x, lengths)
        return x


class LinearDecoderVarLen(nn.Module):
    """
    Linear decoder for variable-length input sequences.

    This module computes the mean of the input tensor along the sequence length dimension for each
    input sequence and applies a linear transformation to the resulting tensor to produce the output.

    Args:
        embedding_size (int): The size of the token embeddings.
        output_size (int): The size of the output tensor.
    """
    def __init__(self, embedding_size, output_size):
        super().__init__()
        self.decoder = nn.Linear(embedding_size, output_size)

    def forward(self, x, lengths=None):
        x = [torch.mean(t, dim=0) for t in torch.split(x, lengths.tolist(), 0)]
        x = torch.stack(x)
        return self.decoder(x)


class LinearDecoder(nn.Module):
    """
    Linear decoder for fixed-length input sequences.

    This module computes the mean of the input tensor along the sequence length dimension and
    applies a linear transformation to the resulting tensor to produce the output.

    Args:
        embedding_size (int): The size of the token embeddings.
        output_size (int): The size of the output tensor.
    """
    def __init__(self, embedding_size, output_size):
        super().__init__()
        self.decoder = nn.Linear(embedding_size, output_size)

    def forward(self, x, lengths=None):
        x = torch.mean(x, dim=1)
        return self.decoder(x)
    
    
class ChordMixer(nn.Module):
    """
    The ChordMixer model. Encoder is a stack of ChordMixer blocks. Decoder a global average pooling, followed by a linear layer.

    Args:
        input_size (int): The input size of the embedding layer.
        output_size (int): The output size of the decoder layer.
        embedding_type (str): The type of embedding layer ('sparse' or 'linear').
        decoder (str): The type of decoder layer. We use 'linear'.
        max_seq_len (int): The maximum sequence length in the data.
        track_size (int): The size of tracks to be rotated.
        hidden_size (int): The hidden layer size for the MLPs.
        mlp_dropout (float): The dropout probability for the MLPs.
        layer_dropout (float): The dropout probability for the ChordMixerBlock.
        prenorm (str): The type of normalization for the pre-normalization step.
        norm (str): The type of normalization for the post-normalization step.
        var_len (bool, optional): Whether to use variable-length mode.
    """
    def __init__(self,
        input_size,
        output_size,
        embedding_type,
        decoder,
        max_seq_len,
        track_size,
        hidden_size,
        mlp_dropout,
        layer_dropout,
        prenorm,
        norm,
        var_len=False
        ):
            super().__init__()
            self.max_n_layers = math.ceil(np.log2(max_seq_len))
            embedding_size = int((self.max_n_layers + 1) * track_size)
            if embedding_type == 'sparse':
                self.embedding = nn.Embedding(
                    input_size,
                    embedding_size,
                    padding_idx=0
                ).apply(self._init_weights)
            elif embedding_type == 'linear':
                self.embedding = nn.Linear(
                    input_size,
                    embedding_size
                ).apply(self._init_weights)
                
            self.encoder = ChordMixerEncoder(
                max_seq_len,
                track_size,
                hidden_size,
                mlp_dropout,
                layer_dropout,
                prenorm,
                norm,
                var_len
            ).apply(self._init_weights)
            
            if decoder == 'linear':
                if var_len:
                    self.decoder = LinearDecoderVarLen(embedding_size, output_size).apply(self._init_weights)
                else:
                    self.decoder = LinearDecoder(embedding_size, output_size).apply(self._init_weights)
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = self.encoder(x, lengths)
        x = self.decoder(x, lengths)
        return x



# for retrieval
class ChordMixerNoDec(nn.Module):
    """
    The ChordMixer model for the Retrieval task. Has no Decoder
    """
    def __init__(self,
        input_size,
        output_size,
        embedding_type,
        decoder,
        max_seq_len,
        track_size,
        hidden_size,
        mlp_dropout,
        layer_dropout,
        prenorm,
        norm,
        var_len=False
        ):
            super().__init__()
            self.max_n_layers = math.ceil(np.log2(max_seq_len))
            embedding_size = int((self.max_n_layers + 1) * track_size)
            if embedding_type == 'sparse':
                self.embedding = nn.Embedding(
                    input_size,
                    embedding_size,
                    padding_idx=0
                ).apply(self._init_weights)
            elif embedding_type == 'linear':
                self.embedding = nn.Linear(
                    input_size,
                    embedding_size
                ).apply(self._init_weights)
                
            self.encoder = ChordMixerEncoder(
                max_seq_len,
                track_size,
                hidden_size,
                mlp_dropout,
                layer_dropout,
                prenorm,
                norm,
                var_len
            ).apply(self._init_weights)
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = self.encoder(x, lengths)
        x = torch.mean(x, dim=1)
        return x
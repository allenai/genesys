"""
xLSTM Block Implementation

This module implements the xLSTM block as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The xLSTM block combines either sLSTM or mLSTM with layer normalization,
residual connections, and additional linear projections.

Author: Mudit Bhargava
Date: June 2024
"""

import torch
import torch.nn as nn


"""
mLSTM: Matrix Long Short-Term Memory

This module implements the mLSTM (matrix LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The mLSTM extends the traditional LSTM by using a matrix memory state and exponential gating,
allowing for enhanced storage capacities and improved performance on long-range dependencies.

Author: Mudit Bhargava
Date: June 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class mLSTM(nn.Module):
    """
    mLSTM layer implementation.

    This layer applies multiple mLSTM cells in sequence, with optional dropout between layers.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
        num_layers (int): Number of mLSTM layers.
        dropout (float, optional): Dropout probability between layers. Default: 0.0.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([mLSTMCell(input_size if i == 0 else hidden_size, hidden_size) 
                                     for i in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass of the mLSTM layer.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_state (tuple of Tensors, optional): Initial hidden state. Default: None.

        Returns:
            tuple: Output sequence and final hidden state.
        """
        batch_size, seq_length, _ = input_seq.size()
        
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
        
        outputs = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            for layer_idx, layer in enumerate(self.layers):
                h, C = hidden_state[layer_idx]
                h, C = layer(x, (h, C))
                hidden_state[layer_idx] = (h, C)
                x = self.dropout_layer(h) if layer_idx < self.num_layers - 1 else h
            outputs.append(x)
        
        return torch.stack(outputs, dim=1), hidden_state

    def init_hidden(self, batch_size):
        """Initialize hidden state for all layers."""
        return [(torch.zeros(batch_size, self.hidden_size, device=self.layers[0].weight_ih.device),
                 torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=self.layers[0].weight_ih.device))
                for _ in range(self.num_layers)]

class mLSTMCell(nn.Module):
    """
    mLSTM cell implementation.

    This cell uses a matrix memory state and exponential gating as described in the xLSTM paper.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
    """

    def __init__(self, input_size, hidden_size):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(3 * hidden_size))
        
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)

    def forward(self, input, hx):
        """
        Forward pass of the mLSTM cell.

        Args:
            input (Tensor): Input tensor of shape (batch_size, input_size).
            hx (tuple of Tensors): Previous hidden state and cell state.

        Returns:
            tuple: New hidden state and cell state.
        """
        h, C = hx
        gates = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        
        i, f, o = gates.chunk(3, 1)
        
        i = torch.exp(i)  # Exponential input gate
        f = torch.exp(f)  # Exponential forget gate
        o = torch.sigmoid(o)
        
        q = self.W_q(input)
        k = self.W_k(input)
        v = self.W_v(input)
        
        C = f.unsqueeze(2) * C + i.unsqueeze(2) * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
        h = o * torch.bmm(q.unsqueeze(1), C).squeeze(1)
        
        return h, C
    
"""
sLSTM: Scalar Long Short-Term Memory

This module implements the sLSTM (scalar LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The sLSTM extends the traditional LSTM by using exponential gating and a new memory mixing technique,
allowing for improved performance on various sequence modeling tasks.

Author: Mudit Bhargava
Date: June 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class sLSTM(nn.Module):
    """
    sLSTM layer implementation.

    This layer applies multiple sLSTM cells in sequence, with optional dropout between layers.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
        num_layers (int): Number of sLSTM layers.
        dropout (float, optional): Dropout probability between layers. Default: 0.0.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([sLSTMCell(input_size if i == 0 else hidden_size, hidden_size) 
                                     for i in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass of the sLSTM layer.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_state (tuple of Tensors, optional): Initial hidden state. Default: None.

        Returns:
            tuple: Output sequence and final hidden state.
        """
        batch_size, seq_length, _ = input_seq.size()
        
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
        
        outputs = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            for layer_idx, layer in enumerate(self.layers):
                h, c = hidden_state[layer_idx]
                h, c = layer(x, (h, c))
                hidden_state[layer_idx] = (h, c)
                x = self.dropout_layer(h) if layer_idx < self.num_layers - 1 else h
            outputs.append(x)
        
        return torch.stack(outputs, dim=1), hidden_state

    def init_hidden(self, batch_size):
        """Initialize hidden state for all layers."""
        return [(torch.zeros(batch_size, self.hidden_size, device=self.layers[0].weight_ih.device),
                 torch.zeros(batch_size, self.hidden_size, device=self.layers[0].weight_ih.device))
                for _ in range(self.num_layers)]

class sLSTMCell(nn.Module):
    """
    sLSTM cell implementation.

    This cell uses exponential gating as described in the xLSTM paper.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
    """

    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(4 * hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias)

    def forward(self, input, hx):
        """
        Forward pass of the sLSTM cell.

        Args:
            input (Tensor): Input tensor of shape (batch_size, input_size).
            hx (tuple of Tensors): Previous hidden state and cell state.

        Returns:
            tuple: New hidden state and cell state.
        """
        h, c = hx
        gates = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        
        i, f, g, o = gates.chunk(4, 1)
        
        i = torch.exp(i)  # Exponential input gate
        f = torch.exp(f)  # Exponential forget gate
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c = f * c + i * g
        h = o * torch.tanh(c)
        
        return h, c

class xLSTMBlock(nn.Module):
    """
    xLSTM block implementation.

    This block can use either sLSTM or mLSTM as its core, surrounded by
    normalization, activation, and projection layers.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state in LSTM.
        num_layers (int): Number of LSTM layers.
        dropout (float, optional): Dropout probability. Default: 0.0.
        lstm_type (str, optional): Type of LSTM to use ('slstm' or 'mlstm'). Default: 'slstm'.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, lstm_type="slstm"):
        super(xLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm_type = lstm_type

        if lstm_type == "slstm":
            self.lstm = sLSTM(input_size, hidden_size, num_layers, dropout)
        elif lstm_type == "mlstm":
            self.lstm = mLSTM(input_size, hidden_size, num_layers, dropout)
        else:
            raise ValueError(f"Invalid LSTM type: {lstm_type}")

        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass of the xLSTM block.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_state (tuple of Tensors, optional): Initial hidden state. Default: None.

        Returns:
            tuple: Output sequence and final hidden state.
        """
        lstm_output, hidden_state = self.lstm(input_seq, hidden_state)
        output = self.activation(lstm_output)
        output = self.norm(output)
        output = self.proj(output)
        output = self.dropout_layer(output + input_seq)  # Residual connection
        return output, hidden_state
    
    
class xLSTM(nn.Module):
    """
    xLSTM model implementation.

    This model uses a combination of sLSTM and mLSTM blocks in a residual architecture.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_size (int): Size of the token embeddings.
        hidden_size (int): Size of the hidden state in LSTM blocks.
        num_layers (int): Number of LSTM layers in each block.
        num_blocks (int): Number of xLSTM blocks.
        dropout (float, optional): Dropout probability. Default: 0.0.
        lstm_type (str, optional): Type of LSTM to use ('slstm' or 'mlstm'). Default: 'slstm'.
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                 dropout=0.0, lstm_type="slstm"):
        super(xLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lstm_type = lstm_type

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.blocks = nn.ModuleList([
            xLSTMBlock(embedding_size, hidden_size, num_layers, dropout, lstm_type)
            for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, input_seq, hidden_states=None):
        """
        Forward pass of the xLSTM model.

        Args:
            input_seq (Tensor): Input sequence of token indices.
            hidden_states (list of tuples, optional): Initial hidden states for each block. Default: None.

        Returns:
            tuple: Output logits and final hidden states.
        """
        embedded_seq = self.embedding(input_seq)
        
        if hidden_states is None:
            hidden_states = [None] * self.num_blocks
        
        output_seq = embedded_seq
        for i, block in enumerate(self.blocks):
            output_seq, hidden_states[i] = block(output_seq, hidden_states[i])
        
        output_seq = self.output_layer(output_seq)
        return output_seq, hidden_states
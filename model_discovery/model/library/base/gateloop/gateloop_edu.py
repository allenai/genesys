from flax import linen as nn
import jax.numpy as jnp
from jax.lax import associative_scan, scan
from typing import Optional, Callable
from flax.linen import initializers


class GatedLinearRNN(nn.Module):
    d_model: int
    d_h: int
    input_activation: Optional[Callable] = nn.tanh
    hidden_activation: Optional[Callable] = nn.tanh
    gate_activation: Optional[Callable] = nn.sigmoid
    use_true_recurrence: Optional[bool] = False
    use_tied_gates: Optional[bool] = True

    def setup(self):
        self.n = 3 if self.use_tied_gates else 4
        self.proj = nn.Dense(self.d_h * self.n)
        self.out_proj = nn.Dense(self.d_model)

        args = dict(
            d_h=self.d_h,
            input_activation=self.input_activation,
            hidden_activation=self.hidden_activation,
            gate_activation=self.gate_activation,
            use_tied_gates=self.use_tied_gates
        )

        if self.use_true_recurrence:
            scan_args = {
                "in_axes": 1,  # Assuming operations are applied along the sequence length axis
                "out_axes": 1,  # Assuming outputs are collected along the sequence length axis
                "variable_broadcast": "params",  # Broadcasting parameters across iterations
                "split_rngs": {"params": False},  # Not splitting RNGs by default
            }
            model_class = nn.scan(
                RecurrentScanGLRU,
                **scan_args,
            )
        else:
            model_class = AssociativeScanGLRU
        self.model = model_class(**args)


    def __call__(self, x, *args, carry=None, **kwargs):
        """
        :param      x:      float   (batch_size, seq_len, d_model) required
                    carry:  float   (batch_size, d_h) (optional)
        :return:    h:      float   (batch_size, d_h)
                    y:      float   (batch_size, seq_len, d_model)
        """
        b, _, _ = x.shape
        x = self.proj(x)
        if self.use_true_recurrence is True:
            if carry is None:
                carry = jnp.zeros((b, self.d_h))
            h, y = self.model(carry, x)
        else:
            h, y = self.model(x, carry=carry)
        y = self.out_proj(y)
        return h, y

def binary_operator(e_i, e_j):
    a_i, kv_i = e_i
    a_j, kv_j = e_j
    return a_j * a_i, a_j * kv_i + kv_j

class AssociativeScanGLRU(nn.Module):
    d_h: int
    input_activation: Optional[Callable] = nn.tanh
    hidden_activation: Optional[Callable] = nn.tanh
    gate_activation: Optional[Callable] = nn.sigmoid
    use_tied_gates: Optional[bool] = True

    def __call__(self, x, carry=None):
        """
        :param      h: float (batch_size, d_h)
                    x: float (batch_size, seq_len, d_h * (3 if self.use_tied_gates is True else 4)
        :return:    y: float (batch_size, seq_len, d_h)
        """
        input = self.input_activation(x[:, :, :self.d_h])
        gates = self.gate_activation(x[:, :, self.d_h:])
        if self.use_tied_gates is True:
            input_gate, output_gate = jnp.split(gates, 2, axis=-1)
            forget_gate = 1 - input_gate
        else:
            input_gate, forget_gate, output_gate = jnp.split(gates, 3, axis=-1)
        scan_ins = input * input_gate
        if carry is not None:
            scan_ins = scan_ins.at[:, 0, :].set(scan_ins[:, 0, :] + carry * forget_gate[:, 0, :])
        _, h = associative_scan(binary_operator, (forget_gate, scan_ins), axis=1)
        y = self.hidden_activation(h) * output_gate
        h = h[:, -1, :]
        return h, y

class RecurrentScanGLRU(nn.Module):
    d_h: int
    input_activation: Optional[Callable] = nn.tanh
    hidden_activation: Optional[Callable] = nn.tanh
    gate_activation: Optional[Callable] = nn.sigmoid
    use_tied_gates: Optional[bool] = True

    def setup(self):
        self.recurrent_proj = nn.Dense(
            self.d_h * (3 if self.use_tied_gates is True else 4),
            use_bias=False,
            kernel_init=initializers.zeros
        )
    def __call__(self, h, x):
        """
        :param      h: float (batch_size, d_h)
                    x: float (batch_size, d_h * (3 if self.use_tied_gates is True else 4))
        :return:    y: float (batch_size, d_h)
        """
        recurrent_t = self.recurrent_proj(h)
        input_t = self.input_activation(x[:, :self.d_h] + recurrent_t[:, :self.d_h])
        gates_t = self.gate_activation(x[:, self.d_h:] + recurrent_t[:, self.d_h:])
        if self.use_tied_gates is True:
            input_gate_t, output_gate_t = jnp.split(gates_t, 2, axis=-1)
            forget_gate_t = 1 - input_gate_t
        else:
            input_gate_t, forget_gate_t, output_gate_t = jnp.split(gates_t, 3, axis=-1)
        h = input_t * input_gate_t + h * forget_gate_t
        y = self.hidden_activation(h) * output_gate_t
        return h, y






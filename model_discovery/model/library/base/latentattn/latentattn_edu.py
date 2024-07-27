from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnums=(3, 5))
def causal_latte(Wq, Wk, Wv, H, X, unroll=100):
    """
    Scan implementation of latte.
    B: batch size, H: number of heads, T: sequence length, D: hidden dimension, L: latent dimension
    Args:
        Wq: jnp.array(D, L) - Query weight matrix
        Wk: jnp.array(D, L) - Key weight matrix
        Wv: jnp.array(D, M) - Value weight matrix
        H: int - number of heads
        X: jnp.array(B, T, D) - input
        unroll: int - unroll factor for the loop
    Returns:
        y: jnp.array(B, T, D) - transformed output sequence
    """

    def accumulate(carry, args):
        csum, norm_cumsum, prev_mx = carry
        Qs_t, curr_alph, V_t, c_mx = args

        revert_maxi = jnp.exp(-c_mx + prev_mx)
        add_maxi = jnp.exp(curr_alph - c_mx)

        norm_cumsum = jnp.einsum("BHL,BHL->BHL", norm_cumsum, revert_maxi)
        norm_cumsum += add_maxi

        carry = jnp.einsum("BHLD,BHL->BHLD", csum, revert_maxi)
        carry += jnp.einsum("BHL,BHD->BHLD", add_maxi, V_t)

        y = jnp.einsum("BHL,BHLD->BHD", Qs_t / norm_cumsum, carry)
        return ((carry, norm_cumsum, c_mx), y)

    B, T, D = X.shape
    L = Wk.shape[-1]

    V = jnp.einsum("DM,BTD->TBM", Wv, X).reshape(T, B, H, -1)
    Q = jnp.einsum("DL,BTD->TBL", Wq, X).reshape(T, B, H, -1)
    K = jnp.einsum("DL,BTD->TBL", Wk, X).reshape(T, B, H, -1)
    maxi = jax.lax.cummax(K, axis=0)

    init_alpha = jnp.zeros(shape=(B, H, L // H))
    init_carry = jnp.zeros((B, H, L // H, D // H))
    Qs = jax.nn.softmax(Q, axis=-1)

    _, y = jax.lax.scan(
        accumulate,
        init=(init_carry, init_alpha, K[0]),
        xs=[Qs, K, V, maxi],
        length=T,
        unroll=unroll
    )

    # TBHD -> BTHD
    y = y.transpose(1, 0, 2, 3)
    return y.reshape(B, T, D)

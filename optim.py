# optim.py
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map


def get_adamw_state(params):
    m = tree_map(jnp.zeros_like, params)
    v = tree_map(jnp.zeros_like, params)
    return m, v


def adamw(params, grads, m, v, t, wd, lr=0.001, b1=0.9, b2=0.99, eps=1e-8):
    b1t = b1**t
    b2t = b2**t

    # Update biased first moment estimate
    m = tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)

    # Update biased second raw moment estimate
    v = tree_map(lambda v, g: b2 * v + (1 - b2) * (g**2), v, grads)

    # Compute bias-corrected first moment estimate
    m_hat = tree_map(lambda m: m / (1 - b1t), m)

    # Compute bias-corrected second raw moment estimate
    v_hat = tree_map(lambda v: v / (1 - b2t), v)

    # Update parameters
    params = tree_map(
        lambda p, m_hat, v_hat: p - lr * (m_hat / (jnp.sqrt(v_hat) + eps) + wd * p),
        params,
        m_hat,
        v_hat,
    )

    return params, m, v


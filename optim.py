import jax
import jax.numpy as jnp


def get_adamw_state(params):
    m = jax.tree.map(lambda x: jnp.zeros_like(x).astype(x.dtype), params)
    v = jax.tree.map(lambda x: jnp.zeros_like(x).astype(x.dtype), params)
    return m, v


def adamw(params, grads, m, v, t, wd, lr=0.001, b1=0.9, b2=0.99, eps=1e-8):
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * (g**2), v, grads)

    m_hat = jax.tree.map(lambda m: m / (1 - b1**t), m)
    v_hat = jax.tree.map(lambda v: v / (1 - b2**t), v)

    params = jax.tree.map(
        lambda p, m_hat, v_hat: p - lr * (m_hat / (jnp.sqrt(v_hat) + eps) + wd * p),
        params,
        m_hat,
        v_hat,
    )

    return params, m, v

import jax
import jax.numpy as jnp
from typing import NamedTuple

from jax.sharding import Mesh
from sharding import sharded_map


def get_adamw_state(params: NamedTuple, mesh: Mesh):
    return sharded_map(params, jnp.zeros_like, mesh), sharded_map(
        params, jnp.zeros_like, mesh
    )


def adamw(params, grads, m, v, t, wd, lr=0.001, b1=0.9, b2=0.99, eps=1e-8):
    t += 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, v, grads)
    m_hat = jax.tree.map(lambda m: m / (1 - b1**t), m)
    v_hat = jax.tree.map(lambda v: v / (1 - b2**t), v)
    params = jax.tree.map(
        lambda p, m, v: p - lr * (m / (jnp.sqrt(v) + eps) + wd * p),
        params,
        m_hat,
        v_hat,
    )

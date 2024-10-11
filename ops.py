import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from utils import AxisNames, mesh


def rope(x: Array, theta: int = 10_000):
    _, _, t, d = x.shape
    emb = jnp.arange(t)[:, None] / (theta ** (jnp.arange(0, d, 2)[None, :] / d))
    cos, sin = jnp.tile(jnp.cos(emb), 2), jnp.tile(jnp.sin(emb), 2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return jnp.concat((x1, x2), axis=-1) * cos + jnp.concat((-x2, x1), axis=-1) * sin


def norm(x: Array, w: Array, eps: float = 1e-6) -> Array:
    return w * (x * jax.lax.rsqrt(jax.lax.pow(x, 2).mean(-1, keepdims=True) + eps))


def mlp(x: Array, w1: Array, w2: Array, w3: Array) -> Array:
    return jnp.dot(jax.nn.silu(jnp.dot(x, w1)) * jnp.dot(x, w3), w2)


qkv_spec = P(AxisNames.dp, None, AxisNames.tp, None)
flash_attn = shard_map(
    lambda q, k, v: flash_attention(q, k, v, causal=True),
    mesh,
    in_specs=(qkv_spec, qkv_spec, qkv_spec),
    out_specs=qkv_spec,
    check_rep=False,
)


def attn(x: Array, wq_chd: Array, wk_chd: Array, wv_chd: Array, wo_hdc: Array) -> Array:
    b, l, c = x.shape  # noqa: E741

    q_bhld = jnp.einsum("blc,chd->bhld", x, wq_chd).astype(jnp.float32)
    k_bhld = jnp.einsum("blc,chd->bhld", x, wk_chd).astype(jnp.float32)
    v_bhld = jnp.einsum("blc,chd->bhld", x, wv_chd).astype(jnp.float32)

    q_bhld, k_bhld = rope(q_bhld), rope(k_bhld)

    out_bhld = flash_attn(q_bhld, k_bhld, v_bhld).astype(x.dtype)
    out_blc = jnp.einsum("bhld,hdc->blc", out_bhld, wo_hdc)
    return out_blc

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from functools import partial

from utils import AxisNames, mesh


def precompute_freqs_cis(dim: int, end: int, theta: float = 10_000.0) -> jax.Array:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs)
    return jnp.exp(1j * freqs)


def rope(x: jax.Array, freqs_cis: jax.Array):
    reshape_x = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
    x_ = jax.lax.complex(reshape_x[..., 0], reshape_x[..., 1])
    x_out = x_ * freqs_cis
    x_out = jnp.stack((jnp.real(x_out), jnp.imag(x_out)), axis=-1)
    return x_out.reshape(x.shape).astype(x.dtype)


@partial(jax.jit, static_argnames=("eps"))
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


def attn(
    x: Array,
    wq_chd: Array,
    wk_chd: Array,
    wv_chd: Array,
    wo_hdc: Array,
    freqs_cis: Array,
) -> Array:
    b, l, c = x.shape  # noqa: E741

    q_bhld = jnp.einsum("blc,chd->bhld", x, wq_chd).astype(jnp.float32)
    k_bhld = jnp.einsum("blc,chd->bhld", x, wk_chd).astype(jnp.float32)
    v_bhld = jnp.einsum("blc,chd->bhld", x, wv_chd).astype(jnp.float32)

    q_bhld, k_bhld = rope(q_bhld, freqs_cis), rope(k_bhld, freqs_cis)

    out_bhld = flash_attn(q_bhld, k_bhld, v_bhld).astype(x.dtype)
    out_blc = jnp.einsum("bhld,hdc->blc", out_bhld, wo_hdc)
    return out_blc

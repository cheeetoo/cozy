import jax
from jax import Array
import jax.numpy as jnp
from einops import repeat


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


def attn(x: Array, wq_chd: Array, wk_chd: Array, wv_chd: Array, wo_hdc: Array) -> Array:
    _, l, c = x.shape  # noqa: E741

    q_bhld = jnp.einsum("blc,chd->bhld", x, wq_chd)
    k_bhld = jnp.einsum("blc,chd->bhld", x, wk_chd)
    v_bhld = jnp.einsum("blc,chd->bhld", x, wv_chd)

    q_bhld, k_bhld = rope(q_bhld), rope(k_bhld)

    n_reps = q_bhld.shape[1] // k_bhld.shape[1]
    k_bhld = repeat(k_bhld, "b h l d -> b (h r) l d", r=n_reps)
    v_bhld = repeat(v_bhld, "b h l d -> b (h r) l d", r=n_reps)

    logits_bhlt = jnp.einsum("...ld,...td->...lt", q_bhld, k_bhld) / c
    logits_bhlt = logits_bhlt + jnp.triu(jnp.full((l, l), -jnp.inf), k=1)
    scores_bhlt = jax.nn.softmax(logits_bhlt.astype(jnp.float32), -1).astype(x.dtype)
    out_bhld = jnp.einsum("...lt,...ld->...td", scores_bhlt, v_bhld)
    out_blc = jnp.einsum("bhld,hdc->blc", out_bhld, wo_hdc)
    return out_blc

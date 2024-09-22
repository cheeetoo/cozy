from jax import Array
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class GPTConfig:
    n_vocab: int
    d_model: int
    n_blocks: int
    d_mlp: int
    n_heads: int
    n_kv_heads: int


class LayerParams(NamedTuple):
    norm_attn: Array
    wq_chd: Array
    wk_chd: Array
    wv_chd: Array
    wo_hdc: Array
    norm_mlp: Array
    w1: Array
    w2: Array
    w3: Array


class Params(NamedTuple):
    tok_emb: Array
    layers: LayerParams
    norm: Array
    head: Array


def init_weights(cfg: GPTConfig, key: Array):
    init = jax.nn.initializers.he_normal()
    d_head = cfg.d_model // cfg.n_heads
    k_emb, k_layers, k_head = jax.random.split(key, 3)
    k_wq, k_wk, k_wv, k_wo, k_w1, k_w2, k_w3 = jax.random.split(k_layers, 4)
    with jax.default_device(jax.devices("cpu")[0]):
        layer_params = LayerParams(
            norm_attn=jnp.ones((cfg.n_blocks, cfg.d_model)),
            wq_chd=init(k_wq, ((cfg.d_model, cfg.n_heads, d_head))),
            wk_chd=init(k_wk, ((cfg.d_model, cfg.n_heads, d_head))),
            wv_chd=init(k_wv, ((cfg.d_model, cfg.n_heads, d_head))),
            wo_hdc=init(k_wo, ((cfg.n_heads, d_head, cfg.d_model))),
            norm_mlp=jnp.ones((cfg.n_blocks, cfg.d_model)),
            w1=init(k_w1, ((cfg.d_model, cfg.d_mlp))),
            w2=init(k_w2, ((cfg.d_mlp, cfg.d_model))),
            w3=init(k_w3, ((cfg.d_model, cfg.d_mlp))),
        )
        params = Params(
            tok_emb=init(k_emb, (cfg.n_vocab, cfg.d_model)),
            layers=layer_params,
            norm=jnp.ones((cfg.d_model)),
            head=init(k_head, (cfg.d_model, cfg.n_vocab)),
        )
    return params


def norm(x: Array, w: Array, eps: float = 1e-6) -> Array:
    return w * (x * jax.lax.rsqrt(jax.lax.pow(x, 2).mean(-1, keepdims=True) + eps))


def mlp(x: Array, w1: Array, w2: Array, w3: Array) -> Array:
    return jnp.dot(jax.nn.silu(jnp.dot(x, w1)) * jnp.dot(x, w3), w2)


def attn(x: Array, wq_chd: Array, wk_chd: Array, wv_chd: Array, wo_hdc: Array) -> Array:
    _, l, c = x.shape  # noqa: E741
    q_blhd = jnp.einsum("...c,chd->...hd", x, wq_chd)
    k_blhd = jnp.einsum("...c,chd->...hd", x, wk_chd)
    v_blhd = jnp.einsum("...c,chd->...hd", x, wv_chd)
    logits_bhlt = jnp.einsum("blhd,bthd->bhlt", q_blhd, k_blhd) / jnp.sqrt(c)
    mask = jnp.triu(jnp.full((l, l), -jnp.inf), k=1)
    scores_blht = jax.nn.softmax(logits_bhlt + mask, -1)
    out_bhld = jnp.einsum("blht,blhd->bhtd", scores_blht, v_blhd)
    out_blc = jnp.einsum("bhld,hdc->blc", out_bhld, wo_hdc)
    return out_blc


def transformer(params: Params, toks: Array):
    x = params.tok_emb[toks]

    def f(x: Array, layer: LayerParams):
        h = norm(x, layer.norm_attn)
        x = x + attn(h, layer.wq_chd, layer.wk_chd, layer.wv_chd, layer.wo_hdc)
        h = norm(x, layer.norm_mlp)
        x = x + mlp(x, layer.w1, layer.w2, layer.w3)
        return x, None

    x, _ = jax.lax.scan(f, x, params.layers)
    return norm(x, params.norm) @ params.head

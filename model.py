from jax import Array
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from dataclasses import dataclass
from typing import NamedTuple

from ops import norm, mlp, attn
from utils import AxisNames


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


shardings = Params(
    tok_emb=P(None, AxisNames.tp),
    layers=LayerParams(
        norm_attn=P(None, None),
        wq_chd=P(None, None, None, AxisNames.tp),
        wk_chd=P(None, None, None, AxisNames.tp),
        wv_chd=P(None, None, None, AxisNames.tp),
        wo_hdc=P(None, None, None, AxisNames.tp),
        w1=P(None, None, AxisNames.tp),
        w2=P(None, None, AxisNames.tp),
        w3=P(None, None, AxisNames.tp),
        norm_mlp=P(None, None),
    ),
    norm=P(None),
    head=P(None, AxisNames.tp),
)


def init_weights(cfg: GPTConfig, key: Array, dtype: jnp.dtype = jnp.float32):
    init = jax.nn.initializers.he_normal()
    d_head = cfg.d_model // cfg.n_heads
    k_emb, k_layers, k_head = jax.random.split(key, 3)
    k_wq, k_wk, k_wv, k_wo, k_w1, k_w2, k_w3 = jax.random.split(k_layers, 7)
    with jax.default_device(jax.devices("cpu")[0]):
        layer_params = LayerParams(
            norm_attn=jnp.ones((cfg.n_blocks, cfg.d_model)),
            wq_chd=init(k_wq, ((cfg.n_blocks, cfg.d_model, cfg.n_heads, d_head))),
            wk_chd=init(k_wk, ((cfg.n_blocks, cfg.d_model, cfg.n_heads, d_head))),
            wv_chd=init(k_wv, ((cfg.n_blocks, cfg.d_model, cfg.n_heads, d_head))),
            wo_hdc=init(k_wo, ((cfg.n_blocks, cfg.n_heads, d_head, cfg.d_model))),
            norm_mlp=jnp.ones((cfg.n_blocks, cfg.d_model)),
            w1=init(k_w1, ((cfg.n_blocks, cfg.d_model, cfg.d_mlp))),
            w2=init(k_w2, ((cfg.n_blocks, cfg.d_mlp, cfg.d_model))),
            w3=init(k_w3, ((cfg.n_blocks, cfg.d_model, cfg.d_mlp))),
        )
        params = Params(
            tok_emb=init(k_emb, (cfg.n_vocab, cfg.d_model)),
            layers=layer_params,
            norm=jnp.ones((cfg.d_model)),
            head=init(k_head, (cfg.d_model, cfg.n_vocab)),
        )
    return jax.tree.map(lambda p: p.astype(dtype), params)


def transformer(params: Params, toks: Array, freqs_cis: Array):
    x = params.tok_emb[toks]

    def f(x: Array, layer: LayerParams):
        h = norm(x.astype(jnp.float32), layer.norm_attn).astype(x.dtype)
        x = x + attn(
            h, layer.wq_chd, layer.wk_chd, layer.wv_chd, layer.wo_hdc, freqs_cis
        )
        h = norm(x.astype(jnp.float32), layer.norm_mlp).astype(x.dtype)
        x = x + mlp(x, layer.w1, layer.w2, layer.w3)
        return x, None

    x, _ = jax.lax.scan(f, x, params.layers)
    return norm(x, params.norm) @ params.head

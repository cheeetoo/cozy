import time
from functools import partial

import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from config import AxisNames, mesh
from model import GPTConfig, init_weights, transformer, shardings
from optim import adamw, get_adamw_state
from sharding import shard_params

devices = jax.devices()

n_vocab = 12_000
cfg = GPTConfig(
    n_vocab=n_vocab,
    d_model=1024,
    n_blocks=16,
    n_heads=8,
    n_kv_heads=4,
    d_mlp=4096,
)
params = init_weights(cfg, jax.random.PRNGKey(42), jnp.bfloat16)
params = shard_params(params, shardings, mesh)

m, v = get_adamw_state(params)

inp = jnp.ones((2, 4096), dtype=int)
inp = jax.device_put(inp, NamedSharding(mesh, P(AxisNames.dp, AxisNames.tp)))

params_shardings = jax.tree.map(lambda x: x.sharding, params)


@jax.value_and_grad
def loss_fn(params, inp):
    logits = transformer(params, inp)[:, :-1].astype(jnp.float32)
    pred = jax.nn.softmax(logits, axis=-1)
    labels = jax.nn.one_hot(inp[:, 1:], num_classes=n_vocab)
    loss = -(labels * jnp.log(pred)).sum(-1).mean()
    return loss


@partial(
    jax.jit,
    in_shardings=(  # type: ignore
        params_shardings,
        params_shardings,
        params_shardings,
        inp.sharding,
        None,
    ),
    out_shardings=(params_shardings, params_shardings, params_shardings, None),  # type: ignore
)
def train_step(params, m, v, inp, t):
    loss, grads = loss_fn(params, inp)
    params, m, v = adamw(params, grads, m, v, t, wd=1e-5, lr=3e-4)
    return params, m, v, loss


STEPS = 20


times = []
with mesh:
    for t in range(1, STEPS + 1):
        start_time = time.time()
        params, m, v, loss = train_step(params, m, v, inp, t)
        duration = time.time() - start_time
        if STEPS - t < 5:
            times.append(duration)
        print(f"Step {t}, Loss: {loss}, Time: {duration:.4f}s")

    compiled = train_step.lower(params, m, v, inp, t).compile()
    avgtime = sum(times) / len(times)
    flops = compiled.cost_analysis()[0]["flops"]
    print(f"{flops=}mfu={((flops/1e+12)/avgtime)/360}")

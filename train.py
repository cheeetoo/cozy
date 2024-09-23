import jax
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from functools import partial
import time
import numpy as np

from model import GPTConfig, init_weights, transformer
from sharding import shard_params, get_sharding
from optim import get_adamw_state, adamw

devices = np.array(jax.devices())
mesh = Mesh(devices, ("data",))

cfg = GPTConfig(
    n_vocab=16384,
    d_model=1024,
    n_blocks=24,
    n_heads=16,
    n_kv_heads=16,
    d_mlp=4096,
)
params = init_weights(cfg, jax.random.PRNGKey(42))

params = shard_params(params, mesh)

m, v = get_adamw_state(params)
# m, v = jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params)

# Create and shard the input data
inp = jnp.ones((8, 512), dtype=int)
inp = jax.device_put(inp, get_sharding(inp, mesh))

# Get sharding specifications for params and inp
params_shardings = jax.tree.map(lambda x: x.sharding, params)
inp_sharding = inp.sharding


def loss_fn(params, inp):
    logits = transformer(params, inp)[:, :-1]
    pred = jax.nn.softmax(logits, axis=-1)
    labels = jax.nn.one_hot(inp[:, 1:], num_classes=16384)
    loss = -(labels * jnp.log(pred)).sum(-1).mean()
    return loss


# Define the training step function with jax.jit
@partial(
    jax.jit,
    in_shardings=(
        params_shardings,
        params_shardings,
        params_shardings,
        inp_sharding,
        None,
    ),
    out_shardings=(params_shardings, params_shardings, params_shardings, None),
)
def train_step(params, m, v, inp, t):
    loss, grads = jax.value_and_grad(loss_fn)(params, inp)
    params, m, v = adamw(params, grads, m, v, t, wd=1e-5, lr=3e-4)
    return params, m, v, loss


STEPS = 20

with mesh:
    for t in range(1, STEPS + 1):
        start_time = time.time()
        params, m, v, loss = train_step(params, m, v, inp, t)
        duration = time.time() - start_time
        print(f"Step {t}, Loss: {loss}, Time: {duration:.4f}s")

    compiled = train_step.lower(params, m, v, inp, t).compile()
    flops = compiled.cost_analysis()[0]["flops"]
    print(f"{flops=}")

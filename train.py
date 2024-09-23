# train.py
import jax
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from functools import partial
import time
import numpy as np

from model import GPTConfig, init_weights, transformer
from sharding import shard_params, get_sharding
from optim import get_adamw_state, adamw

# Define the mesh
devices = np.array(jax.devices())
mesh = Mesh(devices, ("data",))

# Initialize model configuration and parameters
cfg = GPTConfig(1024, 1024, 4, 2048, 8, 4)
params = init_weights(cfg, jax.random.PRNGKey(0))

# Shard the parameters across devices
params = shard_params(params, mesh)

# Initialize optimizer states
m, v = get_adamw_state(params)

# Create and shard the input data
inp = jnp.ones((2, 784), dtype=int)
inp = jax.device_put(inp, get_sharding(inp, mesh))

# Get sharding specifications for params and inp
params_shardings = jax.tree_util.tree_map(lambda x: x.sharding, params)
inp_sharding = inp.sharding


# Define the loss function with jax.jit
@partial(
    jax.jit,
    in_shardings=(params_shardings, inp_sharding),
    out_shardings=None,
    static_argnums=(),
)
def loss_fn(params, inp):
    pred = jax.nn.softmax(transformer(params, inp), axis=-1)[:, :-1]
    labels = jax.nn.one_hot(inp, num_classes=1024)[:, 1:]
    loss = -(labels * jnp.log(pred)).sum(axis=-1).mean()
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
    params, m, v = adamw(params, grads, m, v, t, wd=1e-5, lr=0.001)
    return params, m, v, loss


# Training loop
STEPS = 20

with mesh:
    for t in range(1, STEPS + 1):
        start_time = time.time()
        params, m, v, loss = train_step(params, m, v, inp, t)
        duration = time.time() - start_time
        print(f"Step {t}, Loss: {loss}, Time: {duration:.4f}s")


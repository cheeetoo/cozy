import jax
from jax.sharding import Mesh

from model import GPTConfig, init_weights
from sharding import shard_params
from optim import get_adamw_state, adamw

mesh = Mesh(jax.devices(), "data")

cfg = GPTConfig(1024, 512, 4, 2048, 8, 4)

params = init_weights(cfg, jax.random.PRNGKey(0))

params = shard_params(params, mesh)

m, v = get_adamw_state(params, mesh)

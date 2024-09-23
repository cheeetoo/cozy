# sharding.py
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jax.tree_util import tree_map


def get_spec(x):
    axes = (None,) * x.ndim
    if x.size > 2**18:
        axes = (None,) * (x.ndim - 1) + ("data",)
    return P(*axes)


def get_sharding(x, mesh):
    spec = get_spec(x)
    return NamedSharding(mesh, spec)


def shard_params(params, mesh):
    return tree_map(
        lambda x: jax.device_put(x, get_sharding(x, mesh)),
        params,
    )


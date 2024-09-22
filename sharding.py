from collections.abc import Callable
import jax
from jax import Array, NamedSharding
from jax.sharding import PartitionSpec as P, Mesh
from typing import Any, NamedTuple


def shard_map(x: Array, mesh: Mesh) -> NamedSharding:
    axes: tuple[Any, ...] = (None,) * x.ndim
    if x.size > 2**18:
        axes = (None,) * (x.ndim - 1) + ("data",)
    return NamedSharding(mesh, P(*axes))


def shard_params(params: NamedTuple, mesh: Mesh):
    return sharded_map(params, lambda x: x, mesh)


def sharded_map(pytree: NamedTuple, map_fn: Callable, mesh: Mesh) -> NamedTuple:
    return jax.tree.map(lambda x: jax.device_put(map_fn(x), shard_map(x, mesh)), pytree)

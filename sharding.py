import jax
from jax.sharding import PartitionSpec as P, NamedSharding


def get_sharding(x, mesh):
    axes = (None,) * x.ndim
    if x.size > 2**18:
        axes = (None,) * (x.ndim - 1) + ("data",)
    return NamedSharding(mesh, P(*axes))


def shard_params(params, mesh):
    return sharded_map(lambda x: x, params, mesh)


def sharded_map(map_fn, pytree, mesh):
    return jax.tree.map(
        lambda x: jax.device_put(map_fn(x), get_sharding(x, mesh)), pytree
    )

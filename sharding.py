import jax
from jax.sharding import PartitionSpec as P, NamedSharding


def shard_params(params, shardings, mesh):
    return jax.tree.map(
        lambda p, s: jax.device_put(p, NamedSharding(mesh, s)), params, shardings
    )

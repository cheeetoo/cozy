import jax
from jax.sharding import Mesh
import numpy as np


class AxisNames:
    dp = "replicate"
    tp = "data"


devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 4), (AxisNames.dp, AxisNames.tp))


class Module:
    def __init_subclass__(cls):
        super().__init_subclass__()
        cls._field_names = tuple(cls.__annotations__)
        jax.tree_util.register_pytree_with_keys_class(cls)

    def tree_flatten_with_keys(self):
        return [
            (jax.tree_util.GetAttrKey(name), getattr(self, name))
            for name in self._field_names
        ], None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        for name, child in zip(cls._field_names, children):
            setattr(obj, name, child)
        return obj

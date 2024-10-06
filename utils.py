import jax
from jax.sharding import Mesh
import numpy as np


class AxisNames:
    dp = "replicate"
    tp = "data"


devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 4), (AxisNames.dp, AxisNames.tp))

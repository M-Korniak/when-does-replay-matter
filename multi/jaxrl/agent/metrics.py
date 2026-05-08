import functools

import jax
import jax.numpy as jnp
from jax import tree_util

from jaxrl.utils import Model


@functools.partial(jax.jit, static_argnames=('prefix',))
def param_norms(model: Model, prefix: str):
    leaves = tree_util.tree_leaves(model.params)
    l2 = jnp.sqrt(jnp.sum(jnp.array([jnp.sum(jnp.square(p)) for p in leaves])))
    l1 = jnp.sum(jnp.array([jnp.sum(jnp.abs(p)) for p in leaves]))
    return {f'plasticity_{prefix}_l2_norm': l2, f'plasticity_{prefix}_l1_norm': l1}



import jax.numpy as jnp
from jaxtyping import Float, Array
from jax.experimental import sparse


# Losses
def compute_flops(inputs: Float[Array, "batch_size vocab_size"]):
    return jnp.sum(jnp.square(jnp.mean(jnp.abs(inputs), axis=0)))


def compute_L1(inputs: Float[Array, "batch_size vocab_size"]):
    return jnp.sum(jnp.abs(inputs), axis=-1).mean()


@sparse.sparsify
def compute_flops_sparse(inputs: Float[Array, "batch_size vocab_size"]):
    return jnp.sum(jnp.square(jnp.mean(jnp.abs(inputs), axis=0)))


@sparse.sparsify
def compute_L1_sparse(inputs: Float[Array, "batch_size vocab_size"]):
    return jnp.sum(jnp.abs(inputs), axis=-1).mean()


def create_batch(batch):
    return {k: jnp.array(v) for k, v in batch.items()}

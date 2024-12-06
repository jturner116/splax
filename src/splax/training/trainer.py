from typing import Dict
from functools import partial
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jaxtyping import Float, Int, PRNGKeyArray, PyTree
from .losses import compute_flops, compute_L1
from flax.core import freeze, unfreeze
import flax.linen as nn


class TrainState(train_state.TrainState):
    """Extended train state with SPLADE-specific parameters."""

    lambda_d: Float
    lambda_q: Float
    T_d: Int
    T_q: Int


def create_train_state(
    rng: PRNGKeyArray,
    pretrained_model: nn.Module,
    splade_model: nn.Module,
    dummy_batch: Dict,
    tx: optax.GradientTransformation,
    lambda_d: Float,
    lambda_q: Float,
    T_d: Int,
    T_q: Int,
) -> TrainState:
    """Pure function to create initial training state."""
    # Initialize model with dummy input

    variables = splade_model.init(rng, **dummy_batch, train=True)

    variables = unfreeze(variables)
    variables["params"]["model"] = pretrained_model.params
    variables = freeze(variables)

    return TrainState.create(
        apply_fn=splade_model.apply,
        params=variables["params"],
        tx=tx,
        lambda_d=lambda_d,
        lambda_q=lambda_q,
        T_d=T_d,
        T_q=T_q,
    )


@partial(jax.jit, static_argnames=["top_k_doc", "top_k_query"])
def train_step(
    state: TrainState,
    batch: Dict[str, PyTree],
    dropout_rng: PRNGKeyArray,
    top_k_doc: int,
    top_k_query: int,
):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def compute_lambdas(lambda_init, T, step):
        return jnp.minimum(lambda_init, lambda_init * jnp.square((step + 1) / (T + 1)))

    def loss_fn(params):
        # Directly unpack the batch dictionary
        query_input_ids = batch["query_input_ids"]
        query_attention_mask = batch["query_attention_mask"]
        doc_input_ids = batch["doc_input_ids"]
        doc_attention_mask = batch["doc_attention_mask"]
        batch_size = query_input_ids.shape[0]

        # Could turn this into one forward pass since not treating
        # queries and docs differently
        query_embeddings = state.apply_fn(
            {"params": params},
            query_input_ids,
            query_attention_mask,
            top_k=top_k_query,
            train=True,
            rngs={"dropout": new_dropout_rng},
        )

        num_docs = doc_input_ids.shape[1]  # 1 + num_negatives
        doc_input_ids_flat = doc_input_ids.reshape(-1, doc_input_ids.shape[-1])
        doc_attention_mask_flat = doc_attention_mask.reshape(
            -1, doc_attention_mask.shape[-1]
        )

        doc_embeddings = state.apply_fn(
            {"params": params},
            doc_input_ids_flat,
            doc_attention_mask_flat,
            top_k=top_k_doc,
            train=True,
            rngs={"dropout": new_dropout_rng},
        )

        doc_embeddings = doc_embeddings.reshape(batch_size, num_docs, -1)

        # Compute scores between all queries and all documents
        # Shape: (batch_size, batch_size * 2)
        scores = jnp.sum(query_embeddings[:, None, :] * doc_embeddings, axis=-1)

        # Create labels: for each query, its positive doc is at position i*2
        labels = jnp.zeros(batch_size, dtype=jnp.int32)

        lambda_t_d = compute_lambdas(state.lambda_d, state.T_d, state.step)
        lambda_t_q = compute_lambdas(state.lambda_q, state.T_q, state.step)

        flops = lambda_t_d * compute_flops(
            doc_embeddings.reshape(-1, doc_embeddings.shape[-1])
        ) + lambda_t_q * compute_L1(query_embeddings)

        anti_zero = 1 / (jnp.sum(query_embeddings) ** 2 + 1e-8) + 1 / (
            jnp.sum(doc_embeddings) ** 2 + 1e-8
        )

        triplet_loss = optax.softmax_cross_entropy_with_integer_labels(
            scores, labels
        ).mean()

        total_loss = triplet_loss + flops + anti_zero

        metrics = {
            "loss": total_loss,
            "triplet_loss": triplet_loss,
            "flops": flops,
            "anti_zero": anti_zero,
        }
        return total_loss, metrics

    # Compute gradients
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # Update state
    state = state.apply_gradients(grads=grads)
    return state, loss, metrics, new_dropout_rng

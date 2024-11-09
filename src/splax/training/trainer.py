from typing import Callable, Any, Dict
from functools import partial
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from dataclasses import dataclass
from jaxtyping import Float, Int, PRNGKeyArray, PyTree
from .losses import compute_flops, compute_L1
from flax.core import freeze, unfreeze
import flax.linen as nn


@dataclass
class TrainConfig:
    """Immutable training configuration."""

    learning_rate: float = 3e-6
    warmup_steps: int = 1000
    batch_size: int = 8
    lambda_d: Float = 5e-4
    lambda_q: Float = 5e-4
    T_d: Int = 10000
    T_q: Int = 10000
    logging_steps: int = 20
    save_steps: int = 5000


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


@jax.jit
def train_step(state: TrainState, batch: Dict[str, PyTree], dropout_rng: PRNGKeyArray):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def compute_lambdas(lambda_init, T, step):
        return jnp.minimum(lambda_init, lambda_init * jnp.square((step + 1) / (T + 1)))

    def loss_fn(params):
        # Directly unpack the batch dictionary
        query_input_ids = batch["query_input_ids"]
        query_attention_mask = batch["query_attention_mask"]
        doc_input_ids = batch["doc_input_ids"]
        doc_attention_mask = batch["doc_attention_mask"]

        query_embeddings = state.apply_fn(
            {"params": params},
            query_input_ids,
            query_attention_mask,
            top_k=64,
            train=True,
            rngs={"dropout": new_dropout_rng},
        )

        doc_embeddings = state.apply_fn(
            {"params": params},
            doc_input_ids.reshape(-1, doc_input_ids.shape[-1]),
            doc_attention_mask.reshape(-1, doc_attention_mask.shape[-1]),
            top_k=256,
            train=True,
            rngs={"dropout": new_dropout_rng},
        )

        doc_embeddings = doc_embeddings.reshape(
            query_input_ids.shape[0], 2, -1
        )  # Shape: (batch_size, 2, vocab_size)

        docs_transposed = jnp.transpose(doc_embeddings, (0, 2, 1))
        queries_sparse = query_embeddings[:, None, :]
        scores = jnp.matmul(queries_sparse, docs_transposed).squeeze(1)
        labels = jnp.zeros(scores.shape[0], dtype=jnp.int32)

        lambda_t_d = compute_lambdas(state.lambda_d, state.T_d, state.step)
        lambda_t_q = compute_lambdas(state.lambda_q, state.T_q, state.step)

        flops = lambda_t_d * compute_flops(
            doc_embeddings.reshape(-1, doc_embeddings.shape[-1])
        ) + lambda_t_q * compute_L1(query_embeddings)

        anti_zero = 1 / (jnp.sum(query_embeddings) ** 2 + 1e-8) + 1 / (
            jnp.sum(doc_embeddings.reshape(-1, doc_embeddings.shape[-1])) ** 2 + 1e-8
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

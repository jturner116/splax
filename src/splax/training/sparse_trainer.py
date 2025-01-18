from typing import Dict
from functools import partial
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jaxtyping import Float, Int, PRNGKeyArray, PyTree
from .losses import compute_flops_sparse, compute_L1_sparse
from flax.core import freeze, unfreeze
import flax.linen as nn
from jax.experimental import sparse


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
        query_input_ids = batch["query_input_ids"]
        query_attention_mask = batch["query_attention_mask"]
        doc_input_ids = batch["doc_input_ids"]
        doc_attention_mask = batch["doc_attention_mask"]
        batch_size = query_input_ids.shape[0]

        # Combine queries and documents into a single batch
        num_docs = doc_input_ids.shape[1]  # 1 + num_negatives
        doc_input_ids_flat = doc_input_ids.reshape(-1, doc_input_ids.shape[-1])
        doc_attention_mask_flat = doc_attention_mask.reshape(
            -1, doc_attention_mask.shape[-1]
        )

        # Concatenate query and document inputs
        combined_input_ids = jnp.concatenate([query_input_ids, doc_input_ids_flat])
        combined_attention_mask = jnp.concatenate(
            [query_attention_mask, doc_attention_mask_flat]
        )

        # Single forward pass for both queries and documents
        combined_embeddings = state.apply_fn(
            {"params": params},
            combined_input_ids,
            combined_attention_mask,
            top_k=max(top_k_query, top_k_doc),  # Use max of both top_k values
            train=True,
            rngs={"dropout": new_dropout_rng},
        )

        # Split the embeddings back into queries and documents
        query_embeddings = combined_embeddings[:batch_size]
        doc_embeddings = combined_embeddings[batch_size:].reshape(
            batch_size, num_docs, -1
        )

        sparse_query = sparse.BCOO.fromdense(
            query_embeddings, n_batch=1, nse=top_k_query
        )
        doc_sparse = sparse.BCOO.fromdense(doc_embeddings, n_batch=2, nse=top_k_doc)

        def sparse_scores_fn(query, docs):
            def single_query_doc_product(query, doc):
                return sparse.bcoo_dot_general(
                    query, doc, dimension_numbers=(([0], [1]), ([], []))
                )

            batched_dot = jax.vmap(single_query_doc_product)
            return batched_dot(query, docs)

        # Compute sparse scores
        scores = sparse_scores_fn(sparse_query.todense(), doc_sparse)

        # Create labels: for each query, its positive doc is at position 0
        labels = jnp.zeros(batch_size, dtype=jnp.int32)

        lambda_t_d = compute_lambdas(state.lambda_d, state.T_d, state.step)
        lambda_t_q = compute_lambdas(state.lambda_q, state.T_q, state.step)

        # Reshape doc embeddings for flops computation
        doc_embeddings_reshaped = sparse.bcoo_reshape(
            doc_sparse,
            new_sizes=(doc_sparse.shape[0] * doc_sparse.shape[1], doc_sparse.shape[2]),
        )

        # Compute flops and L1 using sparsified functions
        flops = lambda_t_d * compute_flops_sparse(
            doc_embeddings_reshaped
        ) + lambda_t_q * compute_L1_sparse(sparse_query)

        # Compute anti_zero using sparse data
        anti_zero = 1 / (jnp.sum(sparse_query.data) ** 2 + 1e-8) + 1 / (
            jnp.sum(doc_sparse.data) ** 2 + 1e-8
        )

        # Compute triplet loss using dense scores
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
    (loss, metrics), grads = sparse.value_and_grad(loss_fn, has_aux=True)(state.params)
    # Update state
    state = state.apply_gradients(grads=grads)
    return state, loss, metrics, new_dropout_rng

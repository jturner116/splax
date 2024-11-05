import jax
import jax.numpy as jnp
from flax import linen as nn

# from transformers import FlaxXLMRobertaModel, XLMRobertaConfig


def splade_max(logits, attention_mask):
    relu = jax.nn.relu
    activations = jnp.log1p(relu(logits)) * attention_mask[:, :, None]
    values = jnp.max(activations, axis=1)
    return values


def top_k_mask(logits, k):
    top_indices = jax.lax.top_k(logits, k=k)[1]
    mask = jnp.zeros_like(logits)
    batch_indices = jnp.arange(logits.shape[0])[:, None]
    mask = mask.at[batch_indices, top_indices].set(1.0)
    return logits * mask


class DistilBERTSplade(nn.Module):
    """
    DistilBERT-based SPLADe model.
    """

    model: nn.Module

    @nn.compact
    def __call__(self, input_ids, attention_mask, top_k=64, train=False):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            deterministic=not train,
        )
        logits = outputs.logits
        activations = splade_max(logits, attention_mask)
        activations = top_k_mask(activations, k=top_k)
        return activations


class BERTSplade(nn.Module):
    """
    BERT-based models including Co-Condenser.
    """

    model: nn.Module

    @nn.compact
    def __call__(self, input_ids, attention_mask, top_k=64, train=False):
        batch_size, sequence_length = input_ids.shape
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.arange(sequence_length)[None, :].repeat(batch_size, axis=0)
        head_mask = None  # This is the default in the BERT implementation

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            deterministic=not train,
        )
        logits = outputs.logits
        activations = splade_max(logits, attention_mask)
        activations = top_k_mask(activations, k=top_k)
        return activations

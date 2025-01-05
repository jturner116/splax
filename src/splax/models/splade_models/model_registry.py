from .distilbert import DistilBERTSplade
from .bert import BERTSplade
from transformers import FlaxDistilBertForMaskedLM

from transformers import FlaxBertForMaskedLM

# from ..base_models.bert import FlaxBertForMaskedLM
from ..base_models.flash_bert import FlashBertForMaskedLM
from typing import Any, Tuple

MODEL_REGISTRY = {
    "distilbert": {
        "model_class": FlaxDistilBertForMaskedLM,
        "splade_class": DistilBERTSplade,
    },
    "bert": {
        "model_class": FlashBertForMaskedLM,
        "splade_class": BERTSplade,
    },
}


def get_splade_model(
    model_name: str, model_family: str, from_pt: bool
) -> Tuple[Any, Any]:
    """
    Get both the transformer model and SPLADE wrapper based on model family.

    Args:
        model_name: Name of the pretrained model to load
        model_family: Family of the model (e.g., "bert" or "distilbert")

    Returns:
        Tuple of (transformer_model, splade_model)
    """
    if model_family not in MODEL_REGISTRY:
        raise ValueError(
            f"Model family {model_family} not supported. "
            f"Available families: {list(MODEL_REGISTRY.keys())}"
        )

    config = MODEL_REGISTRY[model_family]
    transformer_model = config["model_class"].from_pretrained(
        model_name, from_pt=from_pt
    )
    splade_model = config["splade_class"](transformer_model.module)

    return transformer_model, splade_model

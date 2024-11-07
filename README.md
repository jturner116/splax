# SPLAX - SPLADE training in JAX/Flax

SPLADE training and sparse-retrievers are currently much better developed in the PyTorch ecosystem than in Jax. Drawing from the original SPLADE code and from the great implementation of Neural Cherche, SPLAX uses the FLAX models present in Transformers to train SPLADE models. There are several motivations for a JAX implementation of SPLADE:

- Easier experimentation of SPLADE training on TPUs, which is useful for student researchers accessing Google TRC environments
- Rich JAX ecosystem for training: great profiling, Optax for optimizers (schedule-free AdamW supported), Orbax for checkpointing


## Hackable training example (from splade_train.py)

```python
@hydra.main(config_path="conf", config_name="distilbert_base")
def main(cfg: DictConfig):
    cfg = TrainingConfig(**cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = FlaxDistilBertForMaskedLM.from_pretrained(cfg.model.name)
    splade_model = DistilBERTSplade(model.module)

    dummy_batch = {
        "input_ids": jnp.ones((1, 128), dtype=jnp.int32),
        "attention_mask": jnp.ones((1, 128), dtype=jnp.int32),
    }

    dataset = load_dataset( #Triplet dataset
        "json",
        data_files={"train": cfg.data.train_path},
        split="train",
        encoding="utf-8",
    )

    tx = optax.contrib.schedule_free_adamw(
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
    )

    train_model(
        pretrained_model=model,
        splade_model=splade_model,
        tokenizer=tokenizer,
        cfg=cfg,
        dataset=dataset,
        dummy_batch=dummy_batch,
        tx=tx,
    )
```
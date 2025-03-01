from splax.training.trainer import train_step, create_train_state
from splax.data.dataset import MultipleNegativesCollateFn
from splax.models.splade_models.model_registry import get_splade_model
from transformers import AutoTokenizer
import jax.numpy as jnp
import jax
import optax
import orbax.checkpoint as ocp
import os
from torch.utils.data import DataLoader
import wandb
from datasets import load_dataset
from dataclasses import dataclass
from tqdm import tqdm
import hydra
from omegaconf import DictConfig


@dataclass
class TrainingConfig:
    seed: int
    data: DictConfig
    model: DictConfig
    batch_size: int
    num_negatives: int
    lambda_d: float
    lambda_q: float
    T_d: float
    T_q: float
    top_k_doc: int
    top_k_query: int
    epochs: int
    log_every: int
    optimizer: DictConfig
    checkpoint: DictConfig
    wandb: bool
    wandb_project: str


def train_model(
    pretrained_model, splade_model, tokenizer, cfg, dataset, dummy_batch, tx
):
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(
        rng=init_rng,
        pretrained_model=pretrained_model,
        splade_model=splade_model,
        dummy_batch=dummy_batch,
        tx=tx,
        lambda_d=cfg.lambda_d,
        lambda_q=cfg.lambda_q,
        T_d=cfg.T_d,
        T_q=cfg.T_q,
    )

    dataloader = DataLoader(
        dataset,
        collate_fn=MultipleNegativesCollateFn(
            tokenizer, num_negatives=cfg.num_negatives
        ),
        batch_size=cfg.batch_size,
    )

    # Orbax checkpointing
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=cfg.checkpoint.max_to_keep,
        save_interval_steps=cfg.checkpoint.save_interval_steps,
    )
    # Path must be absolute
    checkpoint_manager = ocp.CheckpointManager(
        os.path.abspath(cfg.checkpoint.checkpoint_path), options=checkpoint_options
    )
    if cfg.wandb:
        wandb.init(
            project=cfg.wandb_project,
            config={
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.optimizer.learning_rate,
                "warmup_steps": cfg.optimizer.warmup_steps,
            },
        )

    rng, dropout_rng = jax.random.split(rng)
    ## Training loop
    for epoch in range(cfg.epochs):
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            state, loss, metrics, dropout_rng = train_step(
                state, batch, dropout_rng, cfg.top_k_doc, cfg.top_k_query
            )

            if cfg.wandb and step % cfg.log_every == 0:
                wandb.log({**metrics}, step=step)
            if cfg.checkpoint.save_interval_steps > 0:
                checkpoint_manager.save(step, args=ocp.args.StandardSave(state))


@hydra.main(config_path="conf", config_name="distilbert_base", version_base=None)
def main(cfg: DictConfig):
    cfg = TrainingConfig(**cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model, splade_model = get_splade_model(
        cfg.model.name, cfg.model.family, cfg.model.from_pt
    )

    dummy_batch = {
        "input_ids": jnp.ones((1, 128), dtype=jnp.int32),
        "attention_mask": jnp.ones((1, 128), dtype=jnp.int32),
    }

    dataset = load_dataset(
        "json",
        data_files={"train": cfg.data.train_path},
        split="train",
        encoding="utf-8",
    )

    tx = optax.contrib.schedule_free_adamw(
        learning_rate=cfg.optimizer.learning_rate,
        warmup_steps=cfg.optimizer.warmup_steps,
    )

    accumulator = optax.MultiSteps(tx, every_k_schedule=cfg.optimizer.every_k_schedule)

    train_model(
        pretrained_model=model,
        splade_model=splade_model,
        tokenizer=tokenizer,
        cfg=cfg,
        dataset=dataset,
        dummy_batch=dummy_batch,
        tx=accumulator,
    )


if __name__ == "__main__":
    main()

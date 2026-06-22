import torch
from torch import nn
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from betohumor.metrics import compute_metrics


class WeightedTrainer(Trainer):
    """
    Trainer con CrossEntropyLoss ponderada por clase, para compensar
    el desbalance del dataset.
    """

    def __init__(self, *args, class_weights=(1.0, 1.58), **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = torch.tensor(self.class_weights).to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class _EpochPrintCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        epoch = metrics.get("epoch", state.epoch)
        eval_loss = metrics.get("eval_loss")
        macro_f1 = metrics.get("eval_macro_f1")
        accuracy = metrics.get("eval_accuracy")
        print(
            f"  Época {epoch:.0f} | val_loss={eval_loss:.4f} | macro_f1={macro_f1:.4f} | accuracy={accuracy:.4f}"
        )


def _build_training_args(
    beto_config, output_dir, seed, learning_rate=None, weight_decay=None, quiet=True
):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=beto_config.num_epochs,
        per_device_train_batch_size=beto_config.batch_size,
        per_device_eval_batch_size=beto_config.batch_size,
        learning_rate=(
            learning_rate if learning_rate is not None else beto_config.learning_rate
        ),
        warmup_ratio=beto_config.warmup_ratio,
        weight_decay=(
            weight_decay if weight_decay is not None else beto_config.weight_decay
        ),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=seed,
        fp16=True,
        disable_tqdm=quiet,
        logging_strategy="no" if quiet else "steps",
        logging_steps=200,
        report_to="none",
    )


def train_model(
    model,
    train_dataset,
    val_dataset,
    beto_config,
    output_dir,
    seed,
    learning_rate=None,
    weight_decay=None,
    class_weights=(1.0, 1.58),
    quiet=True,
):
    args = _build_training_args(
        beto_config, output_dir, seed, learning_rate, weight_decay, quiet=quiet
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=beto_config.early_stopping_patience
        )
    ]
    if quiet:
        callbacks.append(_EpochPrintCallback())

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=callbacks,
    )

    trainer.train()
    return trainer

from transformers import AutoModelForSequenceClassification


def build_beto(beto_config):
    """Modelo que solo cambia la capa de clasificación"""
    model = AutoModelForSequenceClassification.from_pretrained(
        beto_config.base_model,
        num_labels=beto_config.num_labels,
        ignore_mismatched_sizes=True,
    )

    for name, param in model.named_parameters():
        if "classifier" not in name and "pooler" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Baseline — entrenables: {trainable:,} / {total:,}")
    return model

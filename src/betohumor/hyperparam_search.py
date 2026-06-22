from betohumor.dataset import HahaDataset
from betohumor.train import train_model
from betohumor.metrics import get_best_macro_f1_from_history, get_training_history


def run_one(
    model_builder_fn,
    params,
    beto_config,
    data_config,
    df_train,
    df_val,
    tokenizer,
    output_dir,
    seed,
):
    """
    Entrena una sola combinación de hiperparámetros para el modelo

    """
    model = model_builder_fn(params)

    train_dataset = HahaDataset(df_train, tokenizer, data_config)
    val_dataset = HahaDataset(df_val, tokenizer, data_config)

    trainer = train_model(
        model,
        train_dataset,
        val_dataset,
        beto_config,
        output_dir,
        seed,
        learning_rate=params.get("learning_rate"),
        weight_decay=params.get("weight_decay"),
    )

    best_macro_f1 = get_best_macro_f1_from_history(trainer)
    return best_macro_f1, trainer


def run_search(
    model_builder_fn,
    search_grid,
    beto_config,
    data_config,
    df_train,
    df_val,
    tokenizer,
    seed,
    output_dir_prefix="results/checkpoints/search",
):
    """
    Corre toda la grilla de búsqueda.

    """
    results = []
    for params in search_grid:
        run_name = "_".join(f"{k}{v}" for k, v in params.items())
        output_dir = f"{output_dir_prefix}/{run_name}"

        print(f"\n=== {run_name} ===")
        macro_f1, trainer = run_one(
            model_builder_fn,
            params,
            beto_config,
            data_config,
            df_train,
            df_val,
            tokenizer,
            output_dir,
            seed,
        )
        history = get_training_history(trainer)

        results.append(
            {
                "params": params,
                "macro_f1": macro_f1,
                "history": history,
                "run_name": run_name,
            }
        )

    results.sort(key=lambda x: x["macro_f1"], reverse=True)
    return results

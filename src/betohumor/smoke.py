"""
Smoke test rápido: confirma que train_loss se registra correctamente
en el historial (un punto por época, alineado con eval_loss), sin
esperar un entrenamiento completo.

Usa una muestra chica del dataset y pocas épocas, solo para validar
que el plotting funciona bien antes de lanzar el grid search real.

Uso:
    python3 -m betohumor.smoke_test_train
"""
from transformers import AutoTokenizer

from betohumor.config import DataConfig, BetoConfig
from betohumor.utils import set_seed
from betohumor.dataset import load_and_split, HahaDataset
from betohumor.models.beto import build_beto
from betohumor.train import train_model
from betohumor.metrics import get_training_history
from betohumor.plots import plot_training_curves


def main():
    data_config = DataConfig()
    beto_config = BetoConfig(num_epochs=4, batch_size=16)  # batch chico para que haya varios steps por época
    set_seed(data_config.seed)

    df_train, df_val, _ = load_and_split(data_config)

    # Muestra chica para que corra en segundos, no minutos
    df_train_small = df_train.sample(n=200, random_state=data_config.seed)
    df_val_small   = df_val.sample(n=50, random_state=data_config.seed)

    tokenizer = AutoTokenizer.from_pretrained(beto_config.base_model)
    train_dataset = HahaDataset(df_train_small, tokenizer, data_config)
    val_dataset   = HahaDataset(df_val_small,   tokenizer, data_config)

    model = build_beto(beto_config)

    trainer = train_model(
        model, train_dataset, val_dataset, beto_config,
        output_dir="results/checkpoints/smoke_test",
        seed=data_config.seed,
    )

    history = get_training_history(trainer)

    print(f"\nPuntos de train_loss registrados: {len(history['train_loss'])}")
    print(f"Puntos de eval_loss registrados:  {len(history['eval_loss'])}")

    if len(history['train_loss']) == 0:
        print("\nPROBLEMA: train_loss sigue vacío, el logging no está funcionando bien.")
    elif len(history['train_loss']) != len(history['eval_loss']):
        print(f"\nCantidad distinta de puntos: train={len(history['train_loss'])} vs eval={len(history['eval_loss'])}")
    else:
        print("\nOK: train_loss y eval_loss tienen la misma cantidad de puntos.")

    fig = plot_training_curves(history)
    fig.savefig("smoke_test_curve.png", dpi=100)
    print("Curva guardada en: smoke_test_curve.png")

    # Chequeo de que el eje X solo tenga ticks enteros
    ax_loss = fig.axes[0]
    xticks = ax_loss.get_xticks()
    no_enteros = [t for t in xticks if t != int(t)]
    if no_enteros:
        print(f"\nPROBLEMA: el eje X tiene ticks no enteros: {no_enteros}")
    else:
        print(f"\nOK: el eje X solo tiene ticks enteros: {list(xticks)}")


if __name__ == "__main__":
    main()
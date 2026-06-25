import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        history["train_epochs"],
        history["train_loss"],
        marker="o",
        color="#4C72B0",
        label="Train Loss",
        alpha=0.7,
    )
    axes[0].plot(
        history["eval_epochs"],
        history["eval_loss"],
        marker="o",
        color="#DD8452",
        label="Validation Loss",
    )
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Curva de pérdida — Train vs Val")
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axes[0].legend()

    axes[1].plot(
        history["eval_epochs"],
        history["macro_f1"],
        marker="o",
        color="#4C72B0",
        label="Macro F1",
    )
    axes[1].plot(
        history["eval_epochs"],
        history["accuracy"],
        marker="o",
        color="#55A868",
        label="Accuracy",
    )
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Curvas de métricas (validación)")
    axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    return fig


def plot_confusion_matrix(
    cm, title="Matriz de confusión", labels=("No humor", "Humor"), is_percentage=True
):
    if is_percentage:
        fmt = ".1f"
        cbar_label = "%"
    else:
        is_integer_values = (cm == cm.astype(int)).all()
        fmt = "d" if is_integer_values else ".2f"
        cbar_label = "Cantidad" if is_integer_values else "Proporción"

    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=list(labels),
        yticklabels=list(labels),
        cbar_kws={"label": cbar_label},
    )
    plt.title(title)
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.show()
    return fig


def plot_top_words(
    top_humor,
    top_no_humor,
    title_humor="Top palabras - Humor",
    title_no_humor="Top palabras - No humor",
):
    words_humor, vals_humor = top_humor
    words_no_humor, vals_no_humor = top_no_humor

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(
        list(reversed(words_humor)), list(reversed(vals_humor)), color="#DD8452"
    )
    axes[1].barh(
        list(reversed(words_no_humor)), list(reversed(vals_no_humor)), color="#4C72B0"
    )
    axes[0].set_title(title_humor)
    axes[1].set_title(title_no_humor)
    axes[0].set_xlabel("Coeficiente")
    axes[1].set_xlabel("Coeficiente")
    axes[0].set_ylabel("Palabra / n-grama")
    axes[1].set_ylabel("Palabra / n-grama")
    plt.tight_layout()
    plt.show()
    return fig


def plot_grid_search_comparison(
    labels, macro_f1_values, title="Comparación de configuraciones", ylabel="Macro F1"
):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, macro_f1_values, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    return fig

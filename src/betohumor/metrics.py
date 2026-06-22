import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(eval_pred):
    """Se calculan las metricas accuracy y macrof1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def nb_log_ratio(x, y_i, y):
    p = x[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)


def get_classification_report_df(y_true, y_pred, target_names=("No humor", "Humor")):
    report = classification_report(
        y_true, y_pred, target_names=list(target_names), output_dict=True
    )
    df_report = pd.DataFrame(report).transpose().round(3)
    return df_report.loc[
        [target_names[0], target_names[1], "macro avg"],
        ["precision", "recall", "f1-score"],
    ]


def get_confusion_matrix(y_true, y_pred, normalize="true"):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    if normalize is not None:
        cm = cm * 100
    return cm


def get_top_words_nbsvm(vectorizer, clf, nb_ratio, top_n=15):
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0] * nb_ratio.A1

    top_humor_idx = np.argsort(coefs)[-top_n:][::-1]
    top_no_humor_idx = np.argsort(coefs)[:top_n]

    top_humor = (feature_names[top_humor_idx], coefs[top_humor_idx])
    top_no_humor = (feature_names[top_no_humor_idx], coefs[top_no_humor_idx])
    return top_humor, top_no_humor


def get_best_macro_f1_from_history(trainer):
    eval_entries = [e for e in trainer.state.log_history if "eval_macro_f1" in e]
    return max(e["eval_macro_f1"] for e in eval_entries)


def get_training_history(trainer):
    history = trainer.state.log_history

    train_epochs, train_loss = [], []
    eval_data = {}  # epoch redondeada -> {eval_loss, macro_f1, accuracy}

    for entry in history:
        if "loss" in entry and "eval_loss" not in entry:
            train_loss.append(entry["loss"])
            train_epochs.append(entry["epoch"])
        if "eval_loss" in entry:
            ep = round(entry["epoch"])
            eval_data[ep] = {
                "eval_loss": entry["eval_loss"],
                "macro_f1": entry["eval_macro_f1"],
                "accuracy": entry["eval_accuracy"],
            }

    eval_epochs = sorted(eval_data.keys())
    eval_loss = [eval_data[e]["eval_loss"] for e in eval_epochs]
    macro_f1 = [eval_data[e]["macro_f1"] for e in eval_epochs]
    accuracy = [eval_data[e]["accuracy"] for e in eval_epochs]

    return {
        "train_epochs": train_epochs,
        "train_loss": train_loss,
        "eval_epochs": eval_epochs,
        "eval_loss": eval_loss,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }

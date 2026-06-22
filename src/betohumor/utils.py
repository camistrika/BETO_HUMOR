import os, random
import numpy as np
import torch
import json


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_model(model, folder_name, output_dir="results"):
    save_path = os.path.join(output_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Modelo guardado en: {save_path}")
    return save_path


def save_history(history, folder_name, output_dir="results"):
    save_path = os.path.join(output_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)
    history_path = os.path.join(save_path, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Historial guardado en: {history_path}")
    return history_path


def save_metrics(metrics, folder_name, output_dir="results"):
    save_path = os.path.join(output_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)
    metrics_path = os.path.join(save_path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas guardadas en: {metrics_path}")
    return metrics_path

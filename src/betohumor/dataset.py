import re
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from betohumor.utils import set_seed
from betohumor.config import DataConfig


def clean_tweet(text: str):
    text = re.sub(r"http\S+|www\.\S+", "", text)  # saca URLs
    text = re.sub(r"@\w+", "", text)               # saca menciones
    text = re.sub(r"\s+", " ", text).strip()       # colapsa espacios
    return text


def load_and_split(data_config: DataConfig):
    set_seed(data_config.seed)

    df = pd.read_csv(data_config.data_path)

    text_col  = data_config.text_col
    label_col = data_config.label_col

    # Limpieza
    df[text_col] = df[text_col].astype(str).apply(clean_tweet)
    df = df.dropna(subset=[text_col, label_col])
    df = df.drop_duplicates(subset=[text_col])
    df[label_col] = df[label_col].astype(int)

    # Split 80/10/10 estratificado
    df_train, df_test = train_test_split(
        df, test_size=0.1,
        stratify=df[label_col],
        random_state=data_config.seed
    )
    df_train, df_val = train_test_split(
        df_train, test_size=0.111,  # ~10% del total
        stratify=df_train[label_col],
        random_state=data_config.seed
    )

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    return df_train, df_val, df_test


class HahaDataset(Dataset):
    """Dataset de PyTorch para tokenizar tweets del corpus HAHA con BETO."""

    def __init__(self, df, tokenizer, data_config):
        self.texts  = df[data_config.text_col].tolist()
        self.labels = df[data_config.label_col].tolist()
        self.tokenizer = tokenizer
        self.max_length = data_config.max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx]),
        }

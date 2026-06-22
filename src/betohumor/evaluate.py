import torch
from torch.utils.data import DataLoader


def predict(model, dataset, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())

    logits = torch.cat(all_logits)
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=-1)
    return preds, probs

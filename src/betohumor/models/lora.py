from transformers import AutoModelForSequenceClassification
from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model


def build_beto_lora(beto_config, lora_config):
    model = AutoModelForSequenceClassification.from_pretrained(
        beto_config.base_model,
        num_labels=beto_config.num_labels,
        ignore_mismatched_sizes=True,
    )
    peft_config = PeftLoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        modules_to_save=["classifier"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

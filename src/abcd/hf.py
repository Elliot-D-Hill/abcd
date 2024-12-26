import pathlib

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torchmetrics.functional.classification import auroc
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from abcd.config import get_config


def make_dataset(df: pd.DataFrame, tokenizer):
    sequences = []
    separator = f" {tokenizer.eos_token} {tokenizer.bos_token} "
    for seq in df["sentence"].to_list():
        sequences.append(separator.join(seq))
    data = {"text": sequences, "label": df["label"].to_list()}
    return Dataset.from_dict(data)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    auc = auroc(logits, labels, task="multiclass", num_classes=logits.size(-1))
    return {"auroc": round(auc.item(), 3) if auc is not None else 0.0}


def run():
    path = pathlib.Path("logs/llm/")
    path.mkdir(parents=True, exist_ok=True)
    for file in path.glob("*"):
        file.unlink()
    cfg = get_config("config.toml", factor_model="within_event", analysis="llm")
    set_seed(cfg.random_seed)
    torch.set_float32_matmul_precision("medium")
    model_dir = "allenai/longformer-base-4096"
    num_labels = 4
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.eos_token_id
    train = pd.read_parquet(cfg.filepaths.data.analytic.train)
    val = pd.read_parquet(cfg.filepaths.data.analytic.val)
    train_dataset = make_dataset(train, tokenizer)
    val_dataset = make_dataset(val, tokenizer)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value", "dense"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        max_steps=2,
        eval_steps=2,
        output_dir="./logs/results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        num_train_epochs=10,
        gradient_accumulation_steps=4,
        weight_decay=0.001,
        fp16=True,
        disable_tqdm=False,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        # torch_compile=True,
        logging_dir="./logs/llm/",
        logging_strategy="steps",
        logging_steps=100,
        report_to=["tensorboard"],
        overwrite_output_dir=True,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        local_rank=-1,
        # group_by_length=True,
        max_grad_norm=1.0,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # resume_from_checkpoint="",
    )
    val_dataset = val_dataset[0:12]  # FIXME remove
    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    trainer.train()
    # trainer.evaluate()

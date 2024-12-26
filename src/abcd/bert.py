import pathlib
from typing import cast

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
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
    data = {"text": sequences, "label": df["y_{t+1}"].astype(int).to_list()}
    return Dataset.from_dict(data)


def run():
    path = pathlib.Path("logs/llm/")
    path.mkdir(parents=True, exist_ok=True)
    for file in path.glob("*"):
        file.unlink()

    cfg = get_config("config.toml", factor_model="within_event", analysis="llm")
    set_seed(cfg.random_seed)
    torch.set_float32_matmul_precision("medium")

    # ------------------------------------------------------
    # 2. Load Model and Tokenizer
    # ------------------------------------------------------
    model_name = "models/bertter"
    output_dir = "logs/bertter_checkpoints"
    checkpoint = True
    if checkpoint:
        model_name = output_dir
    # tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    # ------------------------------------------------------
    # 3. Preprocessing Function
    # ------------------------------------------------------

    dataset_path = pathlib.Path("data/tokenized_dataset")
    if dataset_path.exists():
        tokenized_dataset = load_from_disk(dataset_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def preprocess_function(examples):
            tokenized = tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
            )
            tokenized["label"] = examples["label"]
            return tokenized

        train = pd.read_parquet(cfg.filepaths.data.analytic.train)
        val = pd.read_parquet(cfg.filepaths.data.analytic.val)
        train = make_dataset(train, tokenizer)
        val = make_dataset(val, tokenizer)
        dataset = DatasetDict({"train": train, "validation": val})
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        tokenized_dataset.save_to_disk("data/tokenized_dataset")
    tokenized_dataset = cast(DatasetDict, tokenized_dataset)

    # ------------------------------------------------------
    # 4. Metrics
    # ------------------------------------------------------
    auc_metric = evaluate.load("roc_auc", "multiclass")
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits.astype(np.float32)
        labels = labels.astype(np.int32)
        predictions = np.argmax(logits, axis=-1)
        roc_auc = auc_metric.compute(prediction_scores=logits, references=labels)
        roc_auc = roc_auc["roc_auc"] if roc_auc is not None else 0.0
        accuracy = accuracy_metric.compute(prediction=predictions, references=labels)
        accuracy = accuracy["accuracy"] if accuracy is not None else 0.0
        return {"roc_auc": roc_auc, "accuracy": accuracy}

    # ------------------------------------------------------
    # 5. Training Arguments
    # ------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="logs/llm/",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        torch_compile=True,
        optim="adafactor",
        disable_tqdm=False,
        dataloader_num_workers=16,
    )

    # ------------------------------------------------------
    # 6. Define Trainer
    # ------------------------------------------------------

    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # ------------------------------------------------------
    # 7. Train and Evaluate
    # ------------------------------------------------------

    # trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

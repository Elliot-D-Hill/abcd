from functools import partial

import polars as pl
import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


def add_prompt(df: pl.DataFrame, column: str) -> pl.DataFrame:
    return df.with_columns(
        pl.concat_str(
            pl.lit(column.capitalize()), pl.col(column), separator=": "
        ).alias(column)
    )


def format_variables():
    variables = (
        pl.read_parquet("../data/raw/variable_metadata.parquet")
        .drop_nulls()
        .filter(pl.col("response").ne("Continuous"))
        .with_columns(pl.col("response").str.split(";"))
        .explode("response")
        .with_columns(
            pl.col("response")
            .str.replace("\\s*=\\s*", "=")
            .str.split("=")
            .list.to_struct(fields=["value", "response"])
        )
        .unnest("response")
        .with_columns(pl.col("value").str.replace("\\s+|^$", ""))
        .drop_nulls()
        .with_columns(pl.col("value").cast(pl.Float64).alias("value"))
        .pipe(add_prompt, "dataset")
        .pipe(add_prompt, "respondent")
        .pipe(add_prompt, "question")
        .pipe(add_prompt, "response")
        .with_columns(
            pl.concat_str(
                ["dataset", "respondent", "question", "response"], separator="; "
            ).alias("sentence")
        )
    )
    return variables


def make_nlp_dataset(df: pl.DataFrame, variables: pl.DataFrame) -> pl.DataFrame:
    df = df.drop("split", "acs_raked_propensity_score").unpivot(
        index=["src_subject_id", "eventname", "y_{t+1}"]
    )
    df = df.join(variables, on=["variable", "value"], how="inner")
    df = (
        df.group_by("src_subject_id", "eventname", "y_{t+1}")
        .agg(pl.col("sentence"))
        .sort("src_subject_id", "eventname")
    )
    return df


def make_nlp_data():
    # FIXME replace hard-coded paths
    train = pl.read_parquet("../data/analyses/within_event/llm/analytic/train.parquet")
    val = pl.read_parquet("../data/analyses/within_event/llm/analytic/val.parquet")
    test = pl.read_parquet("../data/analyses/within_event/llm/analytic/test.parquet")
    variables = format_variables()
    train = make_nlp_dataset(train, variables)
    val = make_nlp_dataset(val, variables)
    test = make_nlp_dataset(test, variables)
    return train, val, test


class RandomOrderDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.separator = self.tokenizer.eos_token + " " + self.tokenizer.bos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance: pl.DataFrame = self.data[idx]
        sentences = instance.select(
            sentence=pl.lit(self.tokenizer.bos_token + " ")
            + pl.col("sentence")
            .list.sample(fraction=1, shuffle=True)
            .list.join(separator=self.separator)
            + pl.lit(" " + self.tokenizer.eos_token)
        )["sentence"].item()
        label = instance["y_{t+1}"].to_torch().float()
        return sentences, label


def collate_fn(batch, tokenizer):
    sentences, labels = zip(*batch)
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    labels = torch.concat(labels)
    return inputs, labels


class LLM(LightningModule):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def masked_mean_pool(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        expanded_mask = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embedding = torch.sum(
            token_embeddings * expanded_mask, 1
        ) / torch.clamp(expanded_mask.sum(1), min=1e-9)
        return sentence_embedding

    def forward(self, inputs):
        embeddings = self.model(**inputs)
        embedding = self.masked_mean_pool(embeddings, inputs["attention_mask"])
        logits = self.classifier(embedding)
        return logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)  # (batch_size, num_classes)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)


def train_llm():
    train, val, test = make_nlp_data()
    model_name = "google/bigbird-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, clean_up_tokenization_spaces=True
    )
    dataset = RandomOrderDataset(data=train, tokenizer=tokenizer)
    collate = partial(collate_fn, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    model = AutoModel.from_pretrained(model_name)
    model = LLM(model, num_classes=4)
    trainer = Trainer(max_epochs=1)
    trainer.fit(model, dataloader)

from functools import partial
from typing import Callable

import polars as pl
import torch
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassSensitivityAtSpecificity,
    MulticlassSpecificityAtSensitivity,
)
from torchmetrics.functional import precision_recall_curve, roc
from torchmetrics.wrappers import BootStrapper

from abcd.config import Config
from abcd.dataset import ABCDDataModule
from abcd.model import Network, make_trainer
from abcd.tune import get_model


def make_predictions(cfg: Config, model: Network, data_module: ABCDDataModule):
    trainer = make_trainer(cfg=cfg, checkpoint=False)
    predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())
    outputs, labels = zip(*predictions)
    outputs = torch.concat(outputs).cpu()
    labels = torch.concat(labels).cpu()
    metadata = pl.read_csv(cfg.filepaths.data.raw.metadata)
    test_metadata = metadata.filter(pl.col("Split").eq("test"))
    df = pl.DataFrame({"output": outputs.cpu().numpy(), "label": labels.cpu().numpy()})
    df = pl.concat([test_metadata, df], how="horizontal")
    df = df.with_columns(
        pl.when(pl.col("Quartile at t").eq(4))
        .then(pl.lit("Persistence"))
        .otherwise(pl.lit("Conversion"))
        .alias("High-risk scenario")
    )
    variables = [
        "High-risk scenario",
        "Sex",
        "Race",
        "Age",
        "Follow-up event",
        "ADI quartile",
        "Event year",
    ]
    df = df.unpivot(
        index=["output", "label", "Quartile at t", "Quartile at t+1"],
        on=variables,
        variable_name="Variable",
        value_name="Group",
    )
    return df


def get_predictions(cfg: Config, data_module: ABCDDataModule) -> pl.DataFrame:
    model = get_model(cfg=cfg)
    model.to(cfg.device)
    # FIXME cfg.filepaths.data.results.eval.mkdir(parents=True, exist_ok=True)
    if cfg.predict or not cfg.filepaths.data.results.predictions.is_file():
        df = make_predictions(cfg=cfg, model=model, data_module=data_module)
        df.write_parquet(cfg.filepaths.data.results.predictions)
    else:
        df = pl.read_parquet(cfg.filepaths.data.results.predictions)
    return df


def get_outputs_labels(df: pl.DataFrame):
    outputs = torch.tensor(df["output"].to_list(), dtype=torch.float)
    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    return outputs, labels


def make_curve_df(df, name, quartile, x, y):
    return pl.DataFrame(
        {
            "Metric": name,
            "Variable": df["Variable"][0],
            "Group": df["Group"][0],
            "Quartile at t+1": quartile,
            "x": x,
            "y": y,
        }
    )


def make_curve(df: pl.DataFrame, curve: Callable, name: str):
    outputs, labels = get_outputs_labels(df)
    task = "multiclass"
    num_classes = outputs.shape[-1]
    if name == "ROC":
        x, y, _ = curve(outputs, labels, task=task, num_classes=num_classes)
    if name == "PR":
        y, x, _ = curve(outputs, labels, task=task, num_classes=num_classes)
    dfs = []
    for quartile, (x_i, y_i) in enumerate(zip(x, y), start=1):
        quartile = 4 if outputs.dim() == 1 else quartile
        curve_df = make_curve_df(
            df=df, name=name, quartile=quartile, x=x_i.numpy(), y=y_i.numpy()
        )
        dfs.append(curve_df)
    df = pl.concat(dfs)
    return df


def bootstrap_metric(metric, outputs, labels, n_bootstraps: int):
    bootstrap = BootStrapper(
        metric, num_bootstraps=n_bootstraps, mean=False, std=False, raw=True
    )
    bootstrap.update(outputs, labels)
    bootstraps = bootstrap.compute()["raw"].cpu().numpy()
    columns = [str(i) for i in range(1, outputs.shape[-1] + 1)]
    return pl.DataFrame(bootstraps, schema=columns)


def make_metrics(df: pl.DataFrame, n_bootstraps: int):
    if df.shape[0] < 10:
        return pl.DataFrame(
            {
                "Metric": [],
                "Variable": [],
                "Group": [],
                "Quartile at t+1": [],
                "value": [],
            }
        )
    outputs, labels = get_outputs_labels(df)
    auroc = MulticlassAUROC(num_classes=outputs.shape[-1], average="none")
    ap = MulticlassAveragePrecision(num_classes=outputs.shape[-1], average="none")
    bootstrapped_auroc = bootstrap_metric(
        auroc, outputs, labels, n_bootstraps=n_bootstraps
    ).with_columns(pl.lit("AUROC").alias("Metric"))
    bootstrapped_ap = bootstrap_metric(
        ap, outputs, labels, n_bootstraps=n_bootstraps
    ).with_columns(pl.lit("AP").alias("Metric"))
    df = (
        pl.concat(
            [bootstrapped_auroc, bootstrapped_ap],
            how="diagonal_relaxed",
        )
        .with_columns(
            pl.lit(df["Group"][0]).cast(pl.String).alias("Group"),
            pl.lit(df["Variable"][0]).cast(pl.String).alias("Variable"),
        )
        .melt(id_vars=["Metric", "Variable", "Group"], variable_name="Quartile at t+1")
        .with_columns(pl.col("Quartile at t+1").cast(pl.Int64))
    )
    return df


def calc_sensitivity_and_specificity(df: pl.DataFrame):
    df = df.filter(
        pl.col("Variable").eq("High-risk scenario") & pl.col("Group").eq("Conversion")
    )
    outputs, labels = get_outputs_labels(df=df)
    min_sensitivity = 0.5
    specificity_metric = MulticlassSpecificityAtSensitivity(
        num_classes=outputs.shape[-1], min_sensitivity=min_sensitivity
    )
    specificity, threshold = specificity_metric(outputs, labels)
    specificity_df = pl.DataFrame(
        {
            "Risk": ["None", "Low", "Moderate", "High"],
            "Metric": "Specificity",
            "Minimum": f"Sensitivity >= {min_sensitivity}",
            "Value": specificity.numpy().round(decimals=2),
            "Threshold": threshold[1].numpy().round(decimals=2),
        }
    )
    min_specificity = 0.5
    sensitivity_metric = MulticlassSensitivityAtSpecificity(
        num_classes=outputs.shape[-1], min_specificity=min_specificity
    )
    sensitivity, threshold = sensitivity_metric(outputs, labels)
    sensitivity_df = pl.DataFrame(
        {
            "Risk": ["None", "Low", "Moderate", "High"],
            "Metric": "Sensitivity",
            "Minimum": f"Specificity >= {min_specificity}",
            "Value": sensitivity.numpy().round(decimals=2),
            "Threshold": threshold.numpy().round(decimals=2),
        }
    )
    return pl.concat([specificity_df, sensitivity_df])


def make_prevalence(df: pl.DataFrame):
    return df.with_columns(
        pl.col("Quartile at t+1")
        .count()
        .over("Variable", "Group", "Quartile at t+1")
        .truediv(pl.col("Quartile at t+1").count().over("Variable", "Group"))
        .alias("Prevalence"),
    ).select(["Variable", "Group", "Quartile at t+1", "Prevalence"])


def evaluate_model(cfg: Config, data_module: ABCDDataModule):
    df = get_predictions(cfg=cfg, data_module=data_module)
    df_all = df.filter(pl.col("Variable").eq("High-risk scenario")).with_columns(
        pl.lit("Agnostic").alias("Group")
    )
    df = pl.concat([df_all, df]).drop_nulls("Group")
    grouped_df = df.group_by("Variable", "Group", maintain_order=True)
    prevalence = make_prevalence(df)
    metrics = grouped_df.map_groups(
        partial(make_metrics, n_bootstraps=cfg.evaluation.n_bootstraps)
    )
    metrics = metrics.join(prevalence, on=["Variable", "Group", "Quartile at t+1"])
    metrics.write_parquet(cfg.filepaths.data.results.eval.metrics)
    pr_curve = grouped_df.map_groups(
        partial(make_curve, curve=precision_recall_curve, name="PR")
    )
    roc_curve = grouped_df.map_groups(partial(make_curve, curve=roc, name="ROC"))
    curves = pl.concat([pr_curve, roc_curve], how="diagonal_relaxed").select(
        ["Metric", "Variable", "Group", "Quartile at t+1", "x", "y"]
    )
    curves.write_csv(cfg.filepaths.data.results.eval.curves)
    sens_spec = calc_sensitivity_and_specificity(df=df)
    sens_spec.write_csv(cfg.filepaths.data.results.eval.sens_spec)

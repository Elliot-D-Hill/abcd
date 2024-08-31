from functools import partial
from typing import Callable
import torch
import polars as pl
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassSpecificityAtSensitivity,
    MulticlassSensitivityAtSpecificity,
)
from torchmetrics.wrappers import BootStrapper
from torchmetrics.functional import roc, precision_recall_curve

from abcd.config import Config
from abcd.dataset import ABCDDataModule
from abcd.model import Network, make_trainer


def make_predictions(config: Config, model: Network, data_module: ABCDDataModule):
    trainer = make_trainer(config, checkpoint=False)
    predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())
    outputs, labels = zip(*predictions)
    outputs = torch.concat(outputs)
    labels = torch.concat(labels)
    metadata = pl.read_csv(config.filepaths.data.raw.metadata).with_columns(
        pl.col("Quartile at t", "Quartile at t+1").add(1)
    )
    test_metadata = metadata.filter(pl.col("Split").eq("test"))
    df = pl.DataFrame({"output": outputs.cpu().numpy(), "label": labels.cpu().numpy()})
    return pl.concat([test_metadata, df], how="horizontal")


def get_predictions(df: pl.DataFrame, device: str = "cpu"):
    outputs = torch.tensor(df["output"].to_list(), dtype=torch.float, device=device)
    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long, device=device)
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
    outputs, labels = get_predictions(df)
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


def bootstrap_metric(metric, outputs, labels, n_bootstraps=1000):
    bootstrap = BootStrapper(
        metric, num_bootstraps=n_bootstraps, mean=False, std=False, raw=True
    )
    bootstrap.to(outputs.device)
    bootstrap.update(outputs, labels)
    bootstraps = bootstrap.compute()["raw"].cpu().numpy()
    columns = [str(i) for i in range(1, outputs.shape[-1] + 1)]
    return pl.DataFrame(bootstraps, schema=columns)


def make_metrics(df: pl.DataFrame):
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
    outputs, labels = get_predictions(df)
    auroc = MulticlassAUROC(num_classes=outputs.shape[-1], average="none")
    ap = MulticlassAveragePrecision(num_classes=outputs.shape[-1], average="none")
    bootstrapped_auroc = bootstrap_metric(auroc, outputs, labels).with_columns(
        pl.lit("AUROC").alias("Metric")
    )
    bootstrapped_ap = bootstrap_metric(ap, outputs, labels).with_columns(
        pl.lit("AP").alias("Metric")
    )
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
        pl.col("Variable").eq("Quartile subset") & pl.col("Group").eq("{1,2,3}")
    )
    outputs, labels = get_predictions(df=df)
    specificity = MulticlassSpecificityAtSensitivity(
        num_classes=outputs.shape[-1], min_sensitivity=0.5
    )
    spec = specificity(outputs, labels)
    sensitivity = MulticlassSensitivityAtSpecificity(
        num_classes=outputs.shape[-1], min_specificity=0.5
    )
    sens = sensitivity(outputs, labels)
    spec = pl.DataFrame(
        {
            "Metric": "Specificity",
            "Value": spec[0].numpy().round(decimals=2),
            "Threshold": spec[1].numpy().round(decimals=2),
        }
    )
    sens = pl.DataFrame(
        {
            "Metric": "Sensitivity",
            "Value": sens[0].numpy().round(decimals=2),
            "Threshold": sens[1].numpy().round(decimals=2),
        }
    )
    return pl.concat([spec, sens])


def make_prevalence(df: pl.DataFrame):
    return df.with_columns(
        pl.col("Quartile at t+1")
        .count()
        .over("Variable", "Group", "Quartile at t+1")
        .truediv(pl.col("Quartile at t+1").count().over("Variable"))
        .alias("Prevalence"),
    ).select(["Variable", "Group", "Quartile at t+1", "Prevalence"])


def evaluate_model(data_module: ABCDDataModule, config: Config, model: Network):
    model.to(config.device)
    config.filepaths.data.results.metrics.mkdir(parents=True, exist_ok=True)
    if config.predict or not config.filepaths.data.results.predictions.is_file():
        df = make_predictions(config=config, model=model, data_module=data_module)
        df.write_parquet(config.filepaths.data.results.predictions)
    else:
        df = pl.read_parquet(config.filepaths.data.results.predictions)
    df = df.with_columns(
        pl.when(pl.col("Quartile at t").eq(4))
        .then(pl.lit("{4}"))
        .otherwise(pl.lit("{1,2,3}"))
        .alias("Quartile subset")
    )
    variables = [
        "Quartile subset",
        "Sex",
        "Race",
        "Age",
        "Follow-up event",
        "ADI quartile",
        "Event year",
    ]
    df = df.melt(
        id_vars=["output", "label", "Quartile at t", "Quartile at t+1"],
        value_vars=variables,
        variable_name="Variable",
        value_name="Group",
    )
    df_all = df.filter(pl.col("Variable").eq("Quartile subset")).with_columns(
        pl.lit("{1,2,3,4}").alias("Group")
    )
    df = pl.concat([df_all, df]).drop_nulls("Group")
    grouped_df = df.group_by("Variable", "Group", maintain_order=True)
    prevalence = make_prevalence(df)
    metrics = grouped_df.map_groups(make_metrics)
    metrics = metrics.join(prevalence, on=["Variable", "Group", "Quartile at t+1"])
    metrics.write_csv(config.filepaths.data.results.metrics / "metrics.csv")
    pr_curve = grouped_df.map_groups(
        partial(make_curve, curve=precision_recall_curve, name="PR")
    )
    roc_curve = grouped_df.map_groups(partial(make_curve, curve=roc, name="ROC"))
    curves = pl.concat([pr_curve, roc_curve], how="diagonal_relaxed").select(
        ["Metric", "Variable", "Group", "Quartile at t+1", "x", "y"]
    )
    curves.write_csv(config.filepaths.data.results.metrics / "curves.csv")
    sens_spec = calc_sensitivity_and_specificity(df=df)
    sens_spec.write_csv(
        config.filepaths.data.results.metrics / "sensitivity_specificity.csv"
    )

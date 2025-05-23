from functools import partial
from typing import Callable

import polars as pl
import polars.selectors as cs
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
from abcd.constants import COLUMNS, EVENTS_TO_NAMES
from abcd.dataset import ABCDDataModule
from abcd.trainer import make_trainer
from abcd.tune import get_model


def make_predictions(
    cfg: Config, data_module: ABCDDataModule
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.experiment.analysis == "previous_p_factor":
        df = pl.read_parquet(
            "data/analyses/within_event/previous_p_factor/analytic/metadata.parquet"
        )
        outputs = torch.tensor(df["y_t"].to_dummies().to_numpy()).float()
        labels = torch.tensor(df["y_{t+1}"].to_numpy()).int()
    else:
        model = get_model(cfg=cfg, best=True)
        model.to(cfg.device)
        trainer = make_trainer(cfg=cfg, checkpoint=False)
        predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())
        if predictions is None:
            raise ValueError(
                "trainer.predict returned None. Check your model and dataloader."
            )
        outputs, labels = zip(*predictions)
        outputs = torch.concat(outputs).cpu()
        labels = torch.concat(labels).cpu().int()
        outputs = outputs.permute(0, 2, 1)
        labels = labels.flatten(0, 1)
        outputs = outputs.flatten(0, 1)
        auc = MulticlassAUROC(
            num_classes=outputs.shape[-1],
            average="none",
            ignore_index=cfg.evaluation.ignore_index,
        )
        auroc = auc(outputs, labels)
        print(f"AUROC: {auroc}")
        outputs = outputs.permute(0, 2, 1)
        outputs = outputs.flatten(0, 1)
        labels = labels.flatten(0, 1)
        ignore = labels != cfg.evaluation.ignore_index
        outputs = outputs[ignore]
        labels = labels[ignore]
        pass
    auc = MulticlassAUROC(num_classes=outputs.shape[-1], average="none")
    auroc = auc(outputs, labels)
    print(f"AUROC: {auroc}")
    return outputs, labels


def format_metadata(lf: pl.LazyFrame, cfg: Config) -> pl.LazyFrame:
    lf = (
        lf.with_columns(
            pl.col(cfg.index.event).replace_strict(EVENTS_TO_NAMES, default=None)
        )
        .rename(COLUMNS)
        .with_columns(cs.numeric().cast(pl.Int32))
    )
    return lf.filter(pl.col("Split").eq("test"))


def format_predictions(
    cfg: Config, outputs: torch.Tensor, labels: torch.Tensor
) -> pl.DataFrame:
    metadata = pl.scan_parquet(cfg.filepaths.data.analytic.metadata)
    test_metadata = format_metadata(lf=metadata, cfg=cfg)
    df = pl.LazyFrame(
        {
            "output": outputs.cpu().tolist(),
            "label": labels.cpu().tolist(),
        }
    )
    df = pl.concat([test_metadata, df], how="horizontal")
    df = df.with_columns(pl.col("Quartile at t", "Quartile at t+1").add(1))
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
    return df.collect()


def get_predictions(cfg: Config, data_module: ABCDDataModule) -> pl.DataFrame:
    if cfg.predict or not cfg.filepaths.data.results.predictions.is_file():
        outputs, labels = make_predictions(cfg=cfg, data_module=data_module)
        df = format_predictions(cfg=cfg, outputs=outputs, labels=labels)
        df.write_parquet(cfg.filepaths.data.results.predictions)
    else:
        df = pl.read_parquet(cfg.filepaths.data.results.predictions)
    return df


def get_outputs_labels(df: pl.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = torch.tensor(df["output"].to_list(), dtype=torch.float)
    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.int)
    return outputs, labels


def make_curve_df(df, name, quartile, x, y) -> pl.DataFrame:
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


def make_curve(df: pl.DataFrame, curve: Callable, name: str) -> pl.DataFrame:
    outputs, labels = get_outputs_labels(df)
    task = "multiclass"
    num_classes = outputs.shape[-1]
    if name == "ROC":
        x, y, _ = curve(outputs, labels, task=task, num_classes=num_classes)
    if name == "PR":
        y, x, _ = curve(outputs, labels, task=task, num_classes=num_classes)
    dfs: list[pl.DataFrame] = []
    for quartile, (x_i, y_i) in enumerate(zip(x, y), start=1):
        quartile = 4 if outputs.dim() == 1 else quartile
        curve_df = make_curve_df(
            df=df, name=name, quartile=quartile, x=x_i.numpy(), y=y_i.numpy()
        )
        dfs.append(curve_df)
    df = pl.concat(dfs)
    return df


def bootstrap_metric(metric, outputs, labels, n_bootstraps: int) -> pl.LazyFrame:
    bootstrap = BootStrapper(
        metric, num_bootstraps=n_bootstraps, mean=False, std=False, raw=True
    )
    bootstrap.update(outputs, labels)
    bootstraps = bootstrap.compute()["raw"].cpu().numpy()
    columns = [str(i) for i in range(1, outputs.shape[-1] + 1)]
    return pl.LazyFrame(bootstraps, schema=columns)


def make_metrics(df: pl.DataFrame, n_bootstraps: int) -> pl.DataFrame:
    if df.height < 20:
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
    lf = pl.concat(
        [bootstrapped_auroc, bootstrapped_ap],
        how="diagonal_relaxed",
    )
    group = df["Group"].first()
    variable = df["Variable"].first()
    lf = (
        lf.with_columns(
            pl.lit(group).cast(pl.String).alias("Group"),
            pl.lit(variable).cast(pl.String).alias("Variable"),
        )
        .melt(id_vars=["Metric", "Variable", "Group"], variable_name="Quartile at t+1")
        .with_columns(pl.col("Quartile at t+1").cast(pl.Int32))
    )
    return lf.collect()


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
    df = df.drop_nulls()
    grouped_df = df.group_by("Variable", "Group", maintain_order=True)
    prevalence = make_prevalence(df)
    metrics = grouped_df.map_groups(
        partial(make_metrics, n_bootstraps=cfg.evaluation.n_bootstraps)
    )
    metrics = metrics.join(prevalence, on=["Variable", "Group", "Quartile at t+1"])
    print(metrics)
    metrics.write_parquet(cfg.filepaths.data.results.eval.metrics)
    pr_curve = grouped_df.map_groups(
        partial(make_curve, curve=precision_recall_curve, name="PR")
    )
    roc_curve = grouped_df.map_groups(partial(make_curve, curve=roc, name="ROC"))
    curves = pl.concat([pr_curve, roc_curve], how="diagonal_relaxed").select(
        ["Metric", "Variable", "Group", "Quartile at t+1", "x", "y"]
    )
    curves.write_parquet(cfg.filepaths.data.results.eval.curves)
    sens_spec = calc_sensitivity_and_specificity(df=df)
    sens_spec.write_parquet(cfg.filepaths.data.results.eval.sens_spec)

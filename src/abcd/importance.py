from functools import partial

import polars as pl
import torch
from captum.attr import GradientShap
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

from abcd.config import Config
from abcd.dataset import ABCDDataModule
from abcd.tune import get_model


def predict(x, model):
    output = model(x)
    output = output.sum(dim=1)
    output = output[:, -1]  # 4th (last) quartile output
    return output


def make_shap_values(cfg: Config, data_module: ABCDDataModule):
    model = get_model(cfg, best=True)
    test_batches = data_module.test_dataloader()
    val_batches = data_module.val_dataloader()
    inputs = torch.cat([batch[0] for batch in test_batches])
    background = torch.cat([batch[0] for batch in val_batches])
    inputs = inputs.to("mps:0")
    background = background.to("mps:0")
    model.to("mps:0")
    f = partial(predict, model=model)
    explainer = GradientShap(f)
    shap_values = explainer.attribute(inputs, background)
    inputs = inputs.flatten(0, 1).cpu().numpy()
    shap_values = shap_values.flatten(0, 1).cpu().numpy()
    return shap_values, inputs


def regress_shap_values(df: pl.DataFrame):
    x = df.select("feature_value").to_numpy()
    y = df.select("shap_value").to_numpy()
    return LinearRegression().fit(x, y).coef_.item()


def bootstrap_shap_values(df: pl.DataFrame, n_bootstraps: int):
    data = []
    for _ in range(n_bootstraps):
        resampled = resample(df)  # type: ignore
        coef = regress_shap_values(df=resampled)  # type: ignore
        data.append({"coefficient": coef})
    return pl.DataFrame(data)


def format_shap_values(shap_values, inputs, cfg: Config, columns: list[str]):
    # metadata = pl.read_excel("data/supplement/tables/supplementary_table_1.xlsx")
    metadata = pl.read_parquet("data/supplement/tables/supplementary_table_1.parquet")
    # metadata.write_excel("data/supplement/tables/supplementary_table_1.xlsx")
    shap = pl.DataFrame(shap_values, schema=columns)
    inputs = pl.DataFrame(inputs, schema=columns)
    shap = shap.unpivot(index=cfg.index.sample_id, value_name="shap_value")
    inputs = inputs.unpivot(index=cfg.index.sample_id, value_name="feature_value")
    shap = shap.with_columns(inputs["feature_value"])
    shap = shap.drop_nulls()
    shap = shap.join(metadata, on="variable")
    shap.write_parquet(cfg.filepaths.data.results.shap_values)


def group_shap_values(groups: list[str], cfg: Config):
    df = pl.read_parquet(cfg.filepaths.data.results.shap_values)
    df = df.group_by(cfg.index.sample_id, *groups).agg(
        pl.col("shap_value", "feature_value").sum()
    )
    df.write_parquet(cfg.filepaths.data.results.group_shap_values)


def shap_coef(cfg: Config):
    shap = pl.read_parquet(cfg.filepaths.data.results.shap_values)
    metadata = pl.read_parquet("data/supplement/tables/supplementary_table_1.parquet")
    shap = shap.join(metadata, on=["variable", "respondent"])
    data: list[pl.DataFrame] = []
    groups = ["question", "respondent"]
    for (question, respondent), group in shap.group_by(groups):
        bootstraps = bootstrap_shap_values(group, n_bootstraps=1000)
        bootstraps = bootstraps.with_columns(
            question=pl.lit(question), respondent=pl.lit(respondent)
        )
        data.append(bootstraps)
    df = pl.concat(data)
    df = df.join(metadata, on=groups)
    df.write_parquet(cfg.filepaths.data.results.shap_coef)


def group_shap_coef(groups: list[str], cfg: Config):
    shap = pl.read_parquet(cfg.filepaths.data.results.shap_values)
    shap = shap.group_by(cfg.index.sample_id, *groups).sum()
    data: list[pl.DataFrame] = []
    for (dataset, respondent), group in shap.group_by(groups):
        bootstraps = bootstrap_shap_values(group, n_bootstraps=1000)
        bootstraps = bootstraps.with_columns(
            dataset=pl.lit(dataset), respondent=pl.lit(respondent)
        )
        data.append(bootstraps)
    df = pl.concat(data)
    df.write_parquet(cfg.filepaths.data.results.group_shap_coef)


def summed_group_shap_coef(groups: list[str], cfg: Config):
    shap = pl.read_parquet(cfg.filepaths.data.results.shap_values)
    shap = shap.group_by(cfg.index.sample_id, *groups).sum()
    data: list[pl.DataFrame] = []
    for (dataset, respondent), group in shap.group_by(groups):
        bootstraps = bootstrap_shap_values(group, n_bootstraps=1000)
        bootstraps = bootstraps.with_columns(
            dataset=pl.lit(dataset), respondent=pl.lit(respondent)
        )
        data.append(bootstraps)
    df = pl.concat(data)
    df.write_parquet(cfg.filepaths.data.results.group_shap_coef)


def estimate_importance(cfg: Config, data_module: ABCDDataModule):
    shap_values, inputs = make_shap_values(cfg=cfg, data_module=data_module)
    format_shap_values(shap_values, inputs, cfg=cfg, columns=data_module.columns)
    shap_coef(cfg=cfg)
    groups = ["dataset", "respondent"]
    group_shap_values(groups=groups, cfg=cfg)
    group_shap_coef(groups=groups, cfg=cfg)

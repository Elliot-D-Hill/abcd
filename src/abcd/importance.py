from functools import partial

import polars as pl
import torch
from captum.attr import GradientShap
from lightning import LightningDataModule
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

from abcd.config import Config
from abcd.tune import get_model


def predict(x, model):
    output = model(x)
    output = output.sum(dim=1)
    output = output[:, -1]  # 4th (last) quartile output
    return output


def make_shap_values(cfg: Config, data_module: LightningDataModule):
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
    return shap_values.flatten(0, 1).cpu().numpy()


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


def pad_frame(df: pl.DataFrame, sample_id: str) -> pl.DataFrame:
    dfs = []
    max_len: int = df.group_by(sample_id).agg(pl.len())["len"].max()  # type: ignore
    for group in df.partition_by(sample_id, maintain_order=True):
        padding_length = max_len - group.height
        padding = pl.DataFrame({col: [None] * padding_length for col in df.columns})
        group = group.vstack(padding)
        dfs.append(group)
    return pl.concat(dfs)


def format_shap_values(cfg: Config, data_module: LightningDataModule):
    metadata = pl.read_excel("data/supplement/tables/supplementary_table_1.xlsx")
    features = pl.read_parquet(cfg.filepaths.data.analytic.test)
    features = pad_frame(features, sample_id=cfg.index.sample_id)
    features = features.drop(["split", "y_{t+1}", "acs_raked_propensity_score"])
    sample_id = features[cfg.index.sample_id]
    columns = features.columns[1:]
    features = features.unpivot(index=cfg.index.sample_id, value_name="feature_value")
    shap = make_shap_values(cfg=cfg, data_module=data_module)
    shap = pl.DataFrame(shap, schema=columns).with_columns(sample_id)
    shap = shap.unpivot(index=cfg.index.sample_id, value_name="shap_value")
    shap = shap.join(features, on=[cfg.index.sample_id, "variable"])
    shap = shap.join(metadata, on="variable")
    shap.write_parquet(cfg.filepaths.data.results.shap_values)


def shap_coef(cfg: Config):
    shap = pl.read_parquet(cfg.filepaths.data.results.shap_values)
    metadata = pl.read_excel("data/supplement/tables/supplementary_table_1.xlsx")
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


def group_shap_coef(cfg: Config):
    shap = pl.read_parquet(cfg.filepaths.data.results.shap_values)
    shap = shap.group_by([cfg.index.sample_id, "dataset", "respondent"]).sum()
    data: list[pl.DataFrame] = []
    groups = ["dataset", "respondent"]
    for (dataset, respondent), group in shap.group_by(groups):
        bootstraps = bootstrap_shap_values(group, n_bootstraps=1000)
        bootstraps = bootstraps.with_columns(
            dataset=pl.lit(dataset), respondent=pl.lit(respondent)
        )
        data.append(bootstraps)
    df = pl.concat(data)
    df.write_parquet(cfg.filepaths.data.results.group_shap_coef)


def estimate_importance(cfg: Config, data_module: LightningDataModule):
    make_shap_values(cfg=cfg, data_module=data_module)
    shap_coef(cfg=cfg)
    group_shap_coef(cfg=cfg)

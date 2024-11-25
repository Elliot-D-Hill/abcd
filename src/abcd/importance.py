from functools import partial

import polars as pl
import torch
from captum.attr import GradientShap
from lightning import LightningDataModule
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tqdm import tqdm

from abcd.cfg import Config, get_cfg


def predict(x, model):
    output = model(x)
    return output.sum(dim=1)[:, -1]


def make_shap_values(model, X, background, columns):
    f = partial(predict, model=model.to("mps:0"))
    explainer = GradientShap(f)
    shap_values = explainer.attribute(X.to("mps:0"), background.to("mps:0"))
    return pl.DataFrame(shap_values.flatten(0, 1).cpu().numpy(), schema=columns)


def regress_shap_values(df: pl.DataFrame, groups: list[str] | str, n_bootstraps: int):
    data = []
    for _ in tqdm(range(n_bootstraps)):
        resampled = resample(df)  # type: ignore
        coef = (
            make_pipeline(StandardScaler(), LinearRegression())
            .fit(resampled.select("feature_value"), resampled.select("shap_value"))  # type: ignore
            .named_steps["linearregression"]
            .coef_[0, 0]
        )
        row = df.select(groups)[0].to_dicts()[0] | {"coefficient": coef}
        data.append(row)
    df = pl.DataFrame(data)
    return df


def make_shap(
    cfg: Config,
    model,
    data_module: LightningDataModule,
    analysis: str,
    factor_model: str,
):
    cfg = get_cfg(analysis=analysis, factor_model=factor_model)
    test_dataloader = iter(data_module.test_dataloader())
    X = torch.cat([x for x, _ in test_dataloader])
    val_dataloader = iter(data_module.val_dataloader())
    background = torch.cat([x for x, _ in val_dataloader])
    features = pl.read_csv(cfg.filepaths.data.analytic.test).drop(["y_{t+1}"])
    subject_id = (
        features.select("src_subject_id")
        .unique(subset="src_subject_id", maintain_order=True)
        .select(pl.col("src_subject_id").repeat_by(4).flatten())
        .to_series()
    )
    events = pl.Series([1, 2, 3, 4] * features["src_subject_id"].n_unique()).alias(
        "eventname"
    )
    features.drop_in_place("src_subject_id")
    shap_values = make_shap_values(model, X, background, columns=features.columns)
    metadata = pl.read_csv("data/supplement/tables/supplemental_table_1.csv")
    features = (
        pl.DataFrame(X.flatten(0, 1).numpy(), schema=features.columns)
        .with_columns(subject_id, events)
        .unpivot(index=["src_subject_id", "eventname"], value_name="feature_value")
    )
    shap_values = (
        shap_values.with_columns(subject_id, events)
        .unpivot(index=["src_subject_id", "eventname"], value_name="shap_value")
        .join(features, on=["src_subject_id", "eventname", "variable"], how="inner")
        .join(metadata, on="variable", how="inner")
        .rename({"respondent": "Respondent"})
        .filter(pl.col("dataset").ne("Follow-up event"))
    )
    path = f"data/analyses/{factor_model}/{analysis}/results/"
    shap_values.write_csv(path + "shap_values.csv")

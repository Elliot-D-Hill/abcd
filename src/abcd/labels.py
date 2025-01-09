import polars as pl
import polars.selectors as cs
from nanook.preprocess import filter_null_rows
from nanook.transform import assign_splits
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from abcd.config import Config
from abcd.constants import EVENTS


def label_transformer(cfg: Config):
    pipeline = make_pipeline(
        StandardScaler(),
        SimpleImputer(strategy="mean"),
        FactorAnalysis(n_components=1, random_state=cfg.random_seed),
        KBinsDiscretizer(
            n_bins=cfg.model.output_dim,
            encode="ordinal",
            strategy="quantile",
            random_state=cfg.random_seed,
        ),
    )
    transformers = [("labels", pipeline, cfg.features.mh_p_cbcl.columns)]
    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def apply_transformer(df: pl.DataFrame, cfg: Config, transform) -> pl.DataFrame:
    dfs = []
    for _, event in df.group_by(cfg.index.event):
        train = event.filter(pl.col("split") == "train")
        transformer = transform(cfg=cfg)
        transformer.fit(train)  # type: ignore
        transformed_group = transformer.transform(event)  # type: ignore
        dfs.append(transformed_group)
    df = pl.concat(dfs)
    return df


def shift_y(df: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    shift = (
        pl.col(cfg.index.previous_label)
        .shift(-1)
        .over(cfg.index.sample_id)
        .alias("y_{t+1}")
    )
    df = (
        df.sort(cfg.index.sample_id, pl.col(cfg.index.event).cast(pl.Enum(EVENTS)))
        .with_columns(shift)
        .select(
            cfg.index.split,
            *cfg.index.join_on,
            cfg.index.previous_label,
            cfg.index.label,
        )
        .drop_nulls()
    )
    return df


def make_labels(cfg: Config) -> pl.LazyFrame:
    columns = cfg.index.join_on + cfg.features.mh_p_cbcl.columns
    sites = pl.scan_csv(cfg.filepaths.data.raw.features / "abcd_y_lt.csv").select(
        *cfg.index.join_on, cfg.index.site
    )
    df = (
        pl.scan_csv(cfg.filepaths.data.raw.labels)
        .select(columns)
        .sort(cfg.index.sample_id, pl.col(cfg.index.event).cast(pl.Enum(EVENTS)))
    )
    df = (
        df.join(sites, on=cfg.index.join_on, how="left")
        .sort(cfg.index.sample_id, pl.col(cfg.index.event).cast(pl.Enum(EVENTS)))
        .with_columns(pl.col(cfg.index.site).forward_fill().over(cfg.index.sample_id))
    )
    by = (
        cfg.index.site
        if cfg.experiment.analysis == cfg.index.site
        else cfg.index.sample_id
    )
    df = assign_splits(
        frame=df,
        splits=cfg.preprocess.splits,
        by=by,
        name=cfg.index.split,
    )
    df = filter_null_rows(frame=df, columns=cs.by_name(cfg.features.mh_p_cbcl.columns))
    df = apply_transformer(df=df.collect(), cfg=cfg, transform=label_transformer)
    df = df.rename({"factoranalysis0": cfg.index.previous_label})
    df = shift_y(df=df, cfg=cfg)
    df = df.shrink_to_fit()
    return df.lazy()

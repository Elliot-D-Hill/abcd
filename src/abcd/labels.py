import polars as pl
import polars.selectors as cs
from nanuk.preprocess import filter_null_rows
from nanuk.transform import assign_splits
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


def apply_transformer(df: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    dfs = []
    for _, event in df.group_by(cfg.index.event):
        train = event.filter(pl.col("split") == "train")
        transformer = label_transformer(cfg=cfg)
        transformer.fit(train)  # type: ignore
        transformed_group = transformer.transform(event)  # type: ignore
        dfs.append(transformed_group)
    df = pl.concat(dfs)
    df = df.rename({"factoranalysis0": "y_t"})
    return df


def shift_y(df: pl.LazyFrame, cfg: Config) -> pl.LazyFrame:
    shift = pl.col("y_t").shift(-1).over("src_subject_id").alias("y_{t+1}")
    events = pl.Enum(EVENTS)
    df = (
        df.sort(cfg.index.sample_id, pl.col(cfg.index.event).cast(events))
        .with_columns(shift)
        .drop_nulls()
        .select(cfg.index.split, *cfg.index.join_on, "y_t", cfg.index.label)
        .with_columns(cs.numeric().shrink_dtype())
    )
    return df


def make_labels(cfg: Config) -> pl.LazyFrame:
    columns = cfg.index.join_on + cfg.features.mh_p_cbcl.columns
    df = (
        pl.scan_csv(cfg.filepaths.data.raw.labels)
        .select(columns)
        .filter(pl.col(cfg.index.event).is_in(EVENTS))
    )
    assignment = assign_splits(
        splits=cfg.preprocess.splits,
        group=cfg.experiment.split_on,
        name=cfg.index.split,
    )
    df = df.with_columns(assignment)
    df = filter_null_rows(df=df, columns=cs.by_name(cfg.features.mh_p_cbcl.columns))
    df = apply_transformer(df=df.collect(), cfg=cfg).lazy()
    df = shift_y(df=df, cfg=cfg)
    df = df.with_columns(cs.string().cast(pl.Categorical))
    return df

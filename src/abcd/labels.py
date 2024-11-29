import polars as pl
import polars.selectors as cs
from nanuk.transform import assign_splits
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from tomllib import load

from abcd.config import Config
from abcd.preprocess import filter_null_rows


def label_transformer(cfg: Config):
    pipeline = make_pipeline(
        StandardScaler(),
        SimpleImputer(strategy="mean"),
        FactorAnalysis(n_components=1, random_state=cfg.random_seed),
        KBinsDiscretizer(
            n_bins=cfg.preprocess.n_quantiles,
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


def shift_y(df: pl.DataFrame, join_on: list[str]) -> pl.DataFrame:
    shift = pl.col("y_t").shift(-1).over("src_subject_id").alias("y_{t+1}")
    return (
        df.sort(join_on)
        .with_columns(shift)
        .drop_nulls()
        .select("split", *join_on, "y_t", "y_{t+1}")
    )


def make_splits(cfg: Config):
    train_frac = cfg.preprocess.train_size
    val_test_frac = (1 - train_frac) / 2
    splits = {"train": train_frac, "val": val_test_frac, "test": val_test_frac}
    return assign_splits(splits=splits, group=cfg.index.sample_id, name="split")


Events = pl.Enum(
    [
        "baseline_year_1_arm_1",
        "1_year_follow_up_y_arm_1",
        "2_year_follow_up_y_arm_1",
        "3_year_follow_up_y_arm_1",
        "4_year_follow_up_y_arm_1",
    ]
)


def make_labels(cfg: Config | None = None) -> pl.DataFrame:
    set_config(transform_output="polars")
    if cfg is None:
        with open("cfg.toml", "rb") as f:
            cfg = Config(**load(f))
    columns = cfg.index.join_on + cfg.features.mh_p_cbcl.columns
    df = (
        pl.scan_csv(cfg.filepaths.data.raw.labels)
        .select(columns)
        .with_columns(pl.col(cfg.index.event).cast(Events))
    )
    split_assignment = make_splits(cfg=cfg)
    df = df.with_columns(split_assignment)
    df = filter_null_rows(df=df, columns=cs.by_name(cfg.features.mh_p_cbcl.columns))
    df = apply_transformer(df=df.collect(), cfg=cfg)
    df = shift_y(df=df, join_on=cfg.index.join_on)
    df.write_parquet("data/processed/labels.parquet")
    return df


if __name__ == "__main__":
    make_labels()

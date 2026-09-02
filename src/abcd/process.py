import polars as pl
import polars.selectors as cs
from nanook.frame import assign_splits
from nanook.transform import impute, standardize

from .config import Columns, Config, Filepaths, Index
from .labels import make_labels


def select_features(dataset: pl.DataFrame, columns: Columns) -> pl.DataFrame:
    return dataset.drop(*columns.cbcl, *columns.mri, *columns.excluded)


def transform_features(frame: pl.DataFrame, index: Index) -> pl.DataFrame:
    """Z-score then median-impute every feature with training-split statistics."""
    features = cs.numeric() - cs.by_name(index.event, index.previous_label, index.label)
    train = features.as_expr().filter(pl.col(index.split).eq("train"))
    scaled = standardize(features.as_expr(), method="zscore", train=train)
    imputed = impute(features.as_expr(), method="median", train=train)
    return frame.with_columns(scaled).with_columns(imputed)


def write_splits(
    frame: pl.DataFrame, index: Index, filepaths: Filepaths
) -> dict[str, pl.DataFrame]:
    paths = {"train": filepaths.train, "val": filepaths.val, "test": filepaths.test}
    splits = {}
    for split, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        splits[split] = frame.filter(pl.col(index.split).eq(split)).drop(index.split)
        splits[split].write_parquet(path)
    return splits


def make_dataset(cfg: Config) -> dict[str, pl.DataFrame]:
    """Questions-only features labelled with the next-event p-factor quartile, split by subject."""
    index = cfg.index
    dataset = pl.read_csv(cfg.filepaths.dataset, infer_schema_length=None)
    dataset = assign_splits(
        dataset,
        splits=cfg.preprocess.splits,
        by=index.sample_id,
        name=index.split,
        seed=cfg.random_seed,
    )
    labels = make_labels(
        dataset,
        cbcl=cfg.columns.cbcl,
        index=index,
        n_quartiles=cfg.model.output_dim,
        random_seed=cfg.random_seed,
    )
    features = select_features(dataset, columns=cfg.columns)
    frame = (
        labels.join(features, on=[index.sample_id, index.event], how="left")
        .filter(pl.col(index.event).is_in(cfg.preprocess.feature_events))
        .pipe(transform_features, index=index)
        .sort(index.sample_id, index.event)
    )
    return write_splits(frame, index=index, filepaths=cfg.filepaths)

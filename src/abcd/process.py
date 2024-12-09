from collections import defaultdict
from glob import glob

import numpy as np
import polars as pl
import polars.selectors as cs
from nanook.frame import join_dataframes
from nanook.preprocess import drop_null_columns, filter_null_rows
from nanook.transform import impute, pipeline, standardize

from abcd.config import Config
from abcd.constants import EVENTS, EVENTS_TO_VALUES
from abcd.labels import make_labels


def make_demographics(df: pl.LazyFrame) -> pl.LazyFrame:
    education = cs.by_name("demo_prnt_ed_v2", "demo_prtnr_ed_v2")
    sex = "demo_sex_v2"  # 1 = male, 0 = female
    df = (
        df.with_columns(pl.max_horizontal(education).alias("parent_highest_education"))
        .drop(education)
        .with_columns(pl.all().forward_fill().over("src_subject_id"))
        .with_columns(pl.when(pl.col(sex).eq(1)).then(1).otherwise(0).alias(sex))
    )
    return df


def make_adi(df: pl.LazyFrame) -> pl.LazyFrame:
    adi = cs.by_name(
        "reshist_addr1_adi_perc", "reshist_addr2_adi_perc", "reshist_addr3_adi_perc"
    )
    return df.with_columns(
        pl.mean_horizontal(adi).forward_fill().alias("adi_percentile")
    ).drop(adi)


def select_columns(metadata, cfg: Config) -> pl.Expr:
    columns = (
        cs.by_name(cfg.index.join_on + metadata["columns"])
        if metadata["columns"]
        else cs.all()
    )
    return columns & ~cs.contains(*cfg.preprocess.columns_to_drop)


def identity(df: pl.LazyFrame) -> pl.LazyFrame:
    return df


def get_datasets(cfg: Config, include: list[str] | None = None) -> list[pl.LazyFrame]:
    transforms = defaultdict(lambda: identity)
    transforms["led_l_adi"] = make_adi
    transforms["abcd_p_demo"] = make_demographics
    dfs = []
    files = cfg.features.model_dump().items()
    for filename, metadata in files:
        if (include is not None) and (filename not in include):
            continue
        df = pl.scan_csv(
            source=cfg.filepaths.data.raw.features / f"{filename}.csv",
            null_values=["", "null"],
            infer_schema_length=50_000,
            n_rows=2000 if cfg.fast_dev_run else None,
        )
        columns = select_columns(metadata=metadata, cfg=cfg)
        df = (
            df.select(columns)
            .with_columns(pl.all().replace({777: None, 999: None}))
            .pipe(transforms[filename])
            .filter(pl.col(cfg.index.event).is_in(EVENTS))
            .pipe(filter_null_rows, columns=cs.numeric())
            .pipe(drop_null_columns, cutoff=cfg.preprocess.null_cutoff)
            .with_columns(cs.numeric().shrink_dtype())
        )
        dfs.append(df)
    return dfs


def get_mri_data(cfg: Config) -> list[pl.LazyFrame]:
    files = cfg.filepaths.data.raw.mri.glob("*.csv")
    dfs = []
    for filepath in files:
        df = pl.scan_csv(
            source=filepath,
            null_values=["", "null"],
            infer_schema_length=50_000,
            n_rows=2000 if cfg.fast_dev_run else None,
        )
        df = df.filter(pl.col(cfg.index.event).is_in(EVENTS))
        df = df.pipe(drop_null_columns, cutoff=cfg.preprocess.null_cutoff)
        df = df.with_columns(cs.numeric().shrink_dtype())
        dfs.append(df)
    return dfs


def get_data(cfg: Config) -> list[pl.LazyFrame]:
    if cfg.experiment.analysis == "mri":
        data = get_mri_data(cfg)
    elif cfg.experiment.analysis == "all":
        social = get_datasets(cfg)
        mri = get_mri_data(cfg)
        data = social + mri
    else:
        data = get_datasets(cfg)
    return data


def get_brain_features(cfg: Config):
    brain_datasets = {
        "mri_y_dti_fa_fs_at",
        "mri_y_rsfmr_cor_gp_gp",
        "mri_y_tfmr_sst_csvcg_dsk",
        "mri_y_tfmr_mid_alrvn_dsk",
        "mri_y_tfmr_nback_2b_dsk",
    }
    return [
        column
        for name, features in cfg.features.model_dump().items()
        for column in features["columns"]
        if name in brain_datasets
    ]


def get_features(cfg: Config) -> pl.Expr:
    brain_features = get_brain_features(cfg)
    match cfg.experiment.analysis:
        case "metadata":
            selection = cs.all()
        case "mri":
            selection = cs.all()
        case "questions_mri":
            selection = cs.exclude(cfg.features.mh_p_cbcl.columns)
        case "questions" | "all" | "site" | "time":
            selection = cs.exclude(cfg.features.mh_p_cbcl.columns + brain_features)
        case "questions_symptoms":
            selection = cs.exclude(brain_features)
        case "symptoms" | "autoregressive":
            selection = cs.by_name(cfg.features.mh_p_cbcl.columns)
        case "questions_mri_symptoms":
            selection = cs.all()
        case _:
            raise ValueError(f"Invalid analysis: {cfg.experiment.analysis}")
    exclude = ["race_ethnicity", "interview_date"]
    return selection.exclude(*exclude)


def collect_datasets(cfg: Config) -> pl.LazyFrame:
    dfs = get_data(cfg=cfg)
    df = join_dataframes(frames=dfs, on=cfg.index.join_on, how="left")
    features = get_features(cfg=cfg)
    df = df.select(features).drop_nulls(cfg.index.event)
    return df


def transform_dataset(cfg: Config) -> pl.LazyFrame:
    df = collect_datasets(cfg=cfg)
    labels = make_labels(cfg=cfg)
    df = labels.join(df, on=cfg.index.join_on, how="inner")
    columns = cs.numeric().exclude(cfg.index.label)
    train = columns.filter(pl.col(cfg.index.split).eq("train"))
    scale = standardize(columns, method="zscore", train=train)
    imputation = impute(columns, method="median", train=train)
    transforms = [scale, imputation]
    df = pipeline(df, transforms, over=cfg.index.event)
    df = df.with_columns(
        pl.all()
        .fill_null(strategy="forward")
        .over(cfg.index.sample_id)
        .fill_null(strategy="mean")
    )
    df = df.with_columns(
        pl.col(cfg.index.event).replace_strict(EVENTS_TO_VALUES, default=None)
    ).drop(cs.string() & ~cs.by_name(cfg.index.split, cfg.index.sample_id))
    label_columns = cs.by_name(cfg.index.split, *cfg.index.join_on, cfg.index.label)
    df = df.select(label_columns, cs.exclude(label_columns))
    return df


def make_mri_dataset(cfg: Config, df: pl.DataFrame) -> None:
    by = [cfg.index.split, cfg.index.sample_id]
    for (split, sample), group in df.group_by(by, maintain_order=True):
        label = group.drop_in_place(cfg.index.label).to_numpy().astype(np.float32)
        features = (
            group.drop(cfg.index.split, cfg.index.sample_id)
            .to_numpy()
            .astype(np.float32)
        )
        path = str(cfg.filepaths.data.analytic.path / f"{split}" / f"{sample}.npz")
        np.savez(path, features=features, label=label)


def make_dataset(cfg: Config) -> None:
    df = transform_dataset(cfg=cfg).collect()
    if cfg.experiment.analysis == "mri":
        make_mri_dataset(cfg=cfg, df=df)
    else:
        for split, group in df.group_by(cfg.index.split):
            group.write_parquet(
                cfg.filepaths.data.analytic.path / f"{split[0]}.parquet"
            )


def get_dataset(cfg: Config) -> dict[str, pl.DataFrame | list[str]]:
    if cfg.regenerate:
        make_dataset(cfg=cfg)
    if cfg.experiment.analysis == "mri":
        train = glob(str(cfg.filepaths.data.analytic.path / "train/*.npz"))
        val = glob(str(cfg.filepaths.data.analytic.path / "val/*.npz"))
        test = glob(str(cfg.filepaths.data.analytic.path / "test/*.npz"))
    else:
        train = pl.read_parquet(cfg.filepaths.data.analytic.train)
        val = pl.read_parquet(cfg.filepaths.data.analytic.val)
        test = pl.read_parquet(cfg.filepaths.data.analytic.test)
    return {"train": train, "val": val, "test": test}

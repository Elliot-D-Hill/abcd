from functools import reduce

import polars as pl
import polars.selectors as cs
from nanuk.preprocess import drop_null_columns, filter_null_rows

from abcd.config import Config
from abcd.utils import EVENT_MAPPING, EVENTS


def join_dataframes(dfs: list[pl.LazyFrame], join_on: list[str]) -> pl.LazyFrame:
    return reduce(
        lambda left, right: left.join(
            right,
            how="full",
            coalesce=True,
            on=join_on,
        ),
        dfs,
    )


def make_demographics(df: pl.LazyFrame):
    education = ["demo_prnt_ed_v2", "demo_prtnr_ed_v2"]
    sex = "demo_sex_v2"
    df = (
        df.with_columns(pl.max_horizontal(education).alias("parent_highest_education"))
        .with_columns(pl.all().forward_fill().over("src_subject_id"))
        .with_columns(df.select(sex).collect().to_dummies(sex))
        .drop(education, sex, "demo_sex_v2_null")
    )
    return df


def make_adi(df: pl.LazyFrame):
    adi_columns = [
        "reshist_addr1_adi_perc",
        "reshist_addr2_adi_perc",
        "reshist_addr3_adi_perc",
    ]
    adi = "adi_percentile"
    return df.with_columns(
        pl.mean_horizontal(adi_columns).forward_fill().alias(adi)
    ).drop(adi_columns)


def get_datasets(cfg: Config) -> list[pl.LazyFrame]:
    columns_to_drop = (
        "_nm",
        "_nt",
        "_na",
        "_language",
        "_answered",
        "ss_sbd",
        "ss_da",
        "_total",
        "_mean",
        "sds_",
        "srpf_",
        "_fc",
    )
    dfs = []
    files = cfg.features.model_dump().items()
    for filename, metadata in files:
        df = pl.scan_csv(
            source=cfg.filepaths.data.raw.features / f"{filename}.csv",
            null_values=["", "null"],
            infer_schema_length=50_000,
            n_rows=2000 if cfg.fast_dev_run else None,
        )
        columns = (
            pl.col(cfg.join_on + metadata["columns"])
            if metadata["columns"]
            else df.collect_schema().names()
        )
        df = (
            df.select(columns)
            .select(~cs.contains(*columns_to_drop))
            .with_columns(pl.all().replace({777: None, 999: None}))
            .pipe(filter_null_rows, columns=cs.numeric())
            .pipe(drop_null_columns, cutoff=0.25)
        )
        dfs.append(df)
    return dfs


def get_mri_data(cfg: Config):
    files = cfg.filepaths.data.raw.mri.glob("*.csv")
    dfs = []
    for filepath in files:
        df = (
            pl.scan_csv(
                source=filepath,
                null_values=["", "null"],
                infer_schema_length=50_000,
                n_rows=2000 if cfg.fast_dev_run else None,
            )
            .pipe(filter_null_rows, columns=cs.numeric())
            .pipe(drop_null_columns, cutoff=0.25)
        )
        dfs.append(df)
    return dfs


def generate_data(cfg: Config):
    if cfg.analyses.analysis == "mri":
        datasets = get_mri_data(cfg=cfg)
        df = join_dataframes(dfs=datasets, join_on=cfg.index.join_on)
    else:
        datasets = get_datasets(cfg=cfg)
        df = join_dataframes(dfs=datasets, join_on=cfg.index.join_on)
        df = make_adi(df)
        df = make_demographics(df)
    df = df.filter(pl.col(cfg.index.event).is_in(EVENTS)).with_columns(
        eventname=pl.col(cfg.index.event).replace(EVENT_MAPPING).cast(pl.Int32)
    )
    return df

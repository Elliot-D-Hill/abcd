from pathlib import Path

import polars as pl

from abcd.config import Config


def add_prompt(df: pl.LazyFrame, column: str) -> pl.LazyFrame:
    return df.with_columns(
        pl.concat_str(
            pl.lit(column.capitalize()), pl.col(column), separator=": "
        ).alias(column)
    )


def format_variables(filepath: Path | str) -> pl.LazyFrame:
    df = pl.scan_parquet(filepath)
    df = df.drop_nulls()
    df = df.with_columns(pl.col("response").str.split(";"))
    df = df.explode("response")
    df = df.with_columns(
        pl.col("response")
        .str.replace("\\s*=\\s*", "=")
        .str.split("=")
        .list.to_struct(fields=["value", "response"])
    )
    df = df.unnest("response")
    df = df.with_columns(pl.col("value").str.replace("\\s+|^$", ""))
    df = df.with_columns(pl.col("response").fill_null(value=""))
    df = df.drop_nulls()
    df = df.with_columns(pl.col("value").cast(pl.Float32).alias("value"))
    df = df.pipe(add_prompt, "dataset")
    df = df.pipe(add_prompt, "respondent")
    df = df.pipe(add_prompt, "question")
    df = df.pipe(add_prompt, "response")
    df = df.with_columns(
        pl.concat_str(
            ["dataset", "respondent", "question", "response"], separator="; "
        ).alias("sentence")
    )
    return df


def factorize(df: pl.LazyFrame) -> pl.LazyFrame:
    labels = ["1", "2", "3", "4"]
    factors = (
        pl.col("value")
        .rank()
        .qcut(quantiles=[0.25, 0.5, 0.75], labels=labels, allow_duplicates=True)
        .cast(pl.Float32)
    )
    response = pl.col("response")
    return df.with_columns(
        pl.when(response.str.contains("Continuous"))
        .then(response.str.replace("Continuous", factors))
        .otherwise(response)
        .alias(response.meta.output_name())
    )


def make_nlp_dataset(df: pl.LazyFrame, cfg: Config) -> pl.LazyFrame:
    variables = format_variables(filepath=cfg.filepaths.data.processed.variables)
    idx = cfg.index
    by = [idx.split, idx.sample_id, idx.event, idx.label]
    df = (
        df.drop("acs_raked_propensity_score")
        .unpivot(index=by)
        .drop_nulls()
        .with_columns(pl.col("value").cast(pl.Float32))
    )
    df = df.join(variables, on=["variable", "value"], how="inner")
    df = factorize(df)
    df = df.group_by(by).agg(pl.col("sentence")).sort(by[:-1])
    return df

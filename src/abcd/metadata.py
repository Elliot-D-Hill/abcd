import polars as pl
import polars.selectors as cs
from nanook.frame import join_dataframes

from abcd.config import Config
from abcd.constants import (
    COLUMNS,
    EVENTS,
    EVENTS_TO_NAMES,
    EVENTS_TO_VALUES,
    RACE_MAPPING,
    SEX_MAPPING,
)
from abcd.labels import make_labels
from abcd.process import get_datasets


def make_subject_metadata(cfg: Config, df: pl.LazyFrame) -> pl.LazyFrame:
    sex = pl.col("demo_sex_v2").replace_strict(SEX_MAPPING, default=None)
    race = pl.col("race_ethnicity").replace_strict(RACE_MAPPING, default=None)
    age = pl.col("interview_age").truediv(12).round(0)
    quartiles = ["1", "2", "3", "4"]
    adi = pl.col("adi_percentile").qcut(quantiles=4, labels=quartiles).cast(pl.Int32)
    year = (
        pl.col("interview_date")
        .cast(pl.String)
        .str.to_date(format="%m/%d/%Y")
        .dt.year()
    )
    df = (
        df.with_columns(sex, race, age, adi, year)
        .sort(cfg.index.sample_id, pl.col(cfg.index.event).cast(pl.Enum(EVENTS)))
        .with_columns(
            cs.numeric()
            .cast(pl.Int32)
            .fill_null(strategy="forward")
            .over("src_subject_id")
        )
        .with_columns(cs.numeric().shrink_dtype())
    )
    return df


def make_metadata(cfg: Config) -> None:
    if not cfg.regenerate:
        return
    dfs = get_datasets(cfg=cfg)
    make_variable_metadata(cfg=cfg, dfs=dfs)
    df = join_dataframes(frames=dfs, on=cfg.index.join_on, how="left")
    df = make_subject_metadata(cfg=cfg, df=df)
    labels = make_labels(cfg=cfg)
    df = labels.join(df, on=cfg.index.join_on, how="left").select(COLUMNS.keys())
    df = df.with_columns(
        pl.col(cfg.index.event).replace_strict(EVENTS_TO_VALUES, default=None)
    )
    df.collect().write_parquet(cfg.filepaths.data.processed.metadata)
    df = df.with_columns(
        pl.col(cfg.index.event).replace_strict(EVENTS_TO_NAMES, default=None)
    ).rename(COLUMNS)
    df.collect().write_parquet(cfg.filepaths.data.processed.subject_metadata)


def rename_questions() -> pl.Expr:
    return (
        pl.when(pl.col("variable").str.contains("total_core"))
        .then(pl.lit("Adverse childhood experiences"))
        .when(pl.col("variable").str.contains("adi_percentile"))
        .then(pl.lit("Area deprivation index percentile"))
        .when(pl.col("variable").str.contains("parent_highest_education"))
        .then(pl.lit("Parent highest education"))
        .when(pl.col("variable").str.contains("demo_comb_income_v2"))
        .then(pl.lit("Household income"))
        .when(pl.col("variable").eq(pl.lit("eventname")))
        .then(pl.lit("Follow-up event"))
        .when(pl.col("variable").eq(pl.lit("interview_date")))
        .then(pl.lit("Event year"))
        .when(pl.col("variable").eq(pl.lit("interview_age")))
        .then(pl.lit("Age"))
        .otherwise(pl.col("question"))
        .alias("question")
    )


def rename_datasets() -> pl.Expr:
    return (
        pl.when(pl.col("variable").str.contains("eventname"))
        .then(pl.lit("Follow-up event"))
        .when(pl.col("variable").str.contains("demo_sex_v2_|interview_age"))
        .then(pl.lit("Age and sex"))
        .when(
            pl.col("variable").str.contains(
                "adi_percentile|demo_comb_income_v2|parent_highest_education"
            )
        )
        .then(pl.lit("Socio-economic status"))
        .otherwise(pl.col("dataset"))
        .alias("dataset")
    )


def captialize(column: str) -> pl.Expr:
    return pl.col(column).str.slice(0, 1).str.to_uppercase() + pl.col(column).str.slice(
        1
    )


def format_questions() -> pl.Expr:
    return (
        pl.col("question")
        .str.replace(r"(\d+)\.\s+", "")
        .str.replace(r"\..*|(!s)/(!g).*|\?.*", "")
        .str.to_lowercase()
        .str.slice(0)
    )


def make_variable_df(cfg: Config, columns: list[list[str]]) -> pl.DataFrame:
    dfs: list[pl.DataFrame] = []
    for cols, (filename, metadata) in zip(columns, cfg.features.dict().items()):
        table_metadata = {"table": [], "dataset": [], "respondent": [], "variable": []}
        for column in cols:
            table_metadata["table"].append(filename)
            table_metadata["dataset"].append(metadata["name"])
            table_metadata["respondent"].append(metadata["respondent"])
            table_metadata["variable"].append(column)
            metadata_df = pl.DataFrame(table_metadata)
        dfs.append(metadata_df)
    return pl.concat(dfs)


def remove_spanish(df: pl.DataFrame) -> pl.DataFrame:
    spanish = [
        "\\s*/\\s*[^;]+",
        " Sí",
        " Niego contestar",
        "No lo s√É¬© ",
        " No deseo responder",
        "no hay reglas familiares con respecto al consumo de marihuana fuera de la casa",
        "\\| If Separated",
        "(\\|)",
        "\\.",
    ]
    spanish = "|".join(spanish)
    return df.with_columns(
        pl.col("response").str.replace_all(spanish, "")
    ).with_columns(pl.col("response").str.replace("No No", "No"))


def clean_variables(df: pl.DataFrame):
    sex_question = (
        "What sex was the child assigned at birth, on the original birth certificate?"
    )
    sex_response = "1 = Male; 2 = Female; 3 = Intersex-Male; 4 = Intersex-Female; 999 = Don't know; 777 = Refuse to answer"
    df = (
        df.with_columns(
            pl.when(pl.col("table").eq("mh_p_cbcl"))
            .then(True)
            .otherwise(False)
            .alias("autoregressive"),
            pl.col("variable").replace("demo_sex_v2_1", "demo_sex_v2"),
        )
        .with_columns(
            pl.col("dataset").replace("Problem monitoring", "Parental monitoring"),
            pl.when(pl.col("variable").eq("demo_sex_v2"))
            .then(pl.lit(sex_response))
            .when(pl.col("response").is_null() & pl.col("variable").ne("demo_sex_v2"))
            .then(pl.lit("Continuous"))
            .otherwise(pl.col("response"))
            .alias("response"),
            pl.when(pl.col("variable").eq("demo_sex_v2"))
            .then(pl.lit(sex_question))
            .otherwise(pl.col("question"))
            .alias("question"),
        )
        .filter(pl.col("variable").ne("src_subject_id"))
    )
    return df


def make_variable_metadata(cfg: Config, dfs: list[pl.LazyFrame]) -> None:
    columns = [df.collect_schema().names() for df in dfs]
    variables = make_variable_df(cfg=cfg, columns=columns)
    df = pl.read_csv(
        "data/raw/abcd_data_dictionary.csv",
        columns=["table_name", "var_name", "var_label", "notes"],
    ).rename(
        {
            "table_name": "table",
            "var_name": "variable",
            "var_label": "question",
            "notes": "response",
        }
    )
    df = variables.join(df, on=["table", "variable"], how="left", coalesce=True)
    df = df.with_columns(
        format_questions(), pl.col("dataset").str.replace_all("_", " ")
    )
    df = remove_spanish(df)
    df = clean_variables(df)
    df = df.with_columns(captialize("dataset"), captialize("question"))
    df = df.with_columns(rename_questions(), rename_datasets())
    df = df.unique(subset=["variable"])
    df = df.sort("dataset", "respondent", "variable")
    df.write_parquet("data/raw/variable_metadata.parquet")

import polars as pl
import polars.selectors as cs

from abcd.config import Config, Features
from abcd.process import get_datasets
from abcd.transform import get_dataset


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
        pl.when(pl.col("variable").str.contains("eventname|site_id"))
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


def make_variable_df(dfs: list[pl.LazyFrame], features: Features) -> pl.DataFrame:
    metadata_dfs: list[pl.DataFrame] = []
    for df, (filename, metadata) in zip(dfs, features.model_dump().items()):
        df = df.collect()
        table_metadata = {"table": [], "dataset": [], "respondent": [], "variable": []}
        for column in df.columns:
            table_metadata["table"].append(filename)
            table_metadata["dataset"].append(metadata["name"])
            table_metadata["respondent"].append(metadata["respondent"])
            table_metadata["variable"].append(column)
            metadata_df = pl.DataFrame(table_metadata)
        metadata_dfs.append(metadata_df)
    return pl.concat(metadata_dfs)


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


def make_variable_metadata(dfs: list[pl.LazyFrame], features: Features):
    variables = make_variable_df(dfs=dfs, features=features)
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
    df = (
        df.with_columns(
            format_questions(),
            pl.col("dataset").str.replace_all("_", " "),
            pl.col("response").str.replace_all("\\s*/\\s*[^;]+", ""),
        )
        .with_columns(
            captialize("dataset"),
            captialize("question"),
        )
        .with_columns(
            rename_questions(),
            rename_datasets(),
        )
        .unique(subset=["variable"])
        .sort("dataset", "respondent", "variable")
    )
    df.write_csv("data/raw/variable_metadata.csv")


RACE_MAPPING = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}
SEX_MAPPING = {0: "Female", 1: "Male"}
FOLLOW_UP_MAPPING = {0: "Baseline", 1: "1-year", 2: "2-year", 3: "3-year"}


def make_subject_metadata(splits: dict[str, pl.DataFrame]) -> pl.DataFrame:
    df = pl.concat(
        [
            split.with_columns(pl.lit(name).alias("Split"))
            for name, split in splits.items()
        ]
    )
    rename_mapping = {
        "src_subject_id": "Subject ID",
        "eventname": "Follow-up event",
        "y_t": "Quartile at t",
        "y_{t+1}": "Quartile at t+1",
        "demo_sex_v2_1": "Sex",
        "race_ethnicity": "Race",
        "interview_age": "Age",
        "interview_date": "Event year",
        "adi_percentile": "ADI quartile",
        # "parent_highest_education": "Parent highest education",
        # "demo_comb_income_v2": "Combined income",
    }
    quartiles = ["1", "2", "3", "4"]
    df = (
        df.select("Split", *rename_mapping.keys())
        .rename(rename_mapping)
        .with_columns(
            pl.col("Sex").replace_strict(SEX_MAPPING),
            pl.col("Race").replace_strict(RACE_MAPPING),
            pl.col("Follow-up event")
            .replace_strict(FOLLOW_UP_MAPPING)
            .cast(pl.Enum(["Baseline", "1-year", "2-year", "3-year"])),
            pl.col("Age").truediv(12).round(0).cast(pl.Int32),
            pl.col("ADI quartile")
            .qcut(quantiles=4, labels=quartiles)
            .cast(pl.Enum(quartiles)),
            pl.col("Event year").str.to_date(format="%m/%d/%Y").dt.year(),
            pl.col("Quartile at t", "Quartile at t+1").cast(pl.Int32).add(1),
        )
        .with_columns(cs.numeric().cast(pl.Int32))
    )
    return df


def make_metadata(cfg: Config):
    datasets = get_datasets(cfg=cfg)
    make_variable_metadata(dfs=datasets, features=cfg.features)
    splits = get_dataset(cfg=cfg)
    metadata = make_subject_metadata(splits=splits)
    metadata.write_csv(cfg.filepaths.data.raw.metadata)

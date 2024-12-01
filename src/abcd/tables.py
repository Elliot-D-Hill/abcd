from itertools import product

import polars as pl
import polars.selectors as cs

from abcd.config import Config, get_config
from abcd.constants import GROUP_ORDER, RISK_GROUPS


def cross_tabulation():
    df = pl.read_csv("data/raw/metadata.csv")

    variables = ["Sex", "Age", "Race", "Follow-up event", "Event year", "ADI quartile"]
    columns = ["Variable", "Group", "No risk", "Low risk", "Moderate risk", "High risk"]
    df = (
        df.select(["Quartile at t+1"] + variables)
        .with_columns(pl.col("Quartile at t+1").replace_strict(RISK_GROUPS))
        .unpivot(index="Quartile at t+1", on=variables)
        .group_by("Quartile at t+1", "variable", "value")
        .len()
        .pivot(index=["variable", "value"], on="Quartile at t+1", values="len")
        .rename({"variable": "Variable", "value": "Group"})
        .select(columns)
        .sort(
            "Variable",
            pl.col("Group").cast(
                pl.Enum(["Baseline", "1-year", "2-year", "3-year"]), strict=False
            ),
        )
        .with_columns(pl.sum_horizontal(pl.exclude("Variable", "Group")).alias("Total"))
    )
    col_sum = pl.exclude("Variable", "Group").sum().over("Variable")
    percent = (
        pl.exclude("Variable", "Group")
        .truediv(col_sum)
        .mul(100)
        .round(0)
        .cast(pl.Int32)
        .cast(pl.String)
    )
    df = (
        df.with_columns(
            pl.exclude("Variable", "Group").cast(pl.String) + " (" + percent + "%)"
        )
        .drop_nulls()
        .sort("Variable", pl.col("Group").cast(GROUP_ORDER, strict=False))
    )
    return df


def aggregate_metrics(analyses: list[str], factor_models: list[str]):
    for metric_type in ("sensitivity_specificity", "curves", "metrics"):
        metrics = []
        for analysis, factor_model in product(analyses, factor_models):
            path = f"data/analyses/{factor_model}/{analysis}/results/metrics/{metric_type}.csv"
            metric = (
                pl.scan_csv(path)
                .with_columns(
                    pl.lit(factor_model).alias("Factor model"),
                    pl.lit(analysis).alias("Predictor set"),
                )
                .with_columns(
                    pl.col("Predictor set").replace(
                        {
                            "questions": "Questionnaires",
                            "symptoms": "CBCL scales",
                            "questions_symptoms": "Questionnaires, CBCL scales",
                            "questions_mri": "Questionnaires, MRI",
                            "questions_mri_symptoms": "Questionnaires, MRI, CBCL scales",
                            "autoregressive": "Previous p-factors",
                        }
                    ),
                    pl.col("Factor model").replace(
                        {"within_event": "Within-event", "across_event": "Across-event"}
                    ),
                )
                .with_columns(
                    cs.numeric().shrink_dtype(), cs.string().cast(pl.Categorical)
                )
            )
            metrics.append(metric)
        pl.concat(metrics).collect().write_parquet(
            f"data/results/metrics/{metric_type}.parquet"
        )


def make_metric_table(df: pl.LazyFrame, groups: list[str]):
    (
        df.group_by(groups + ["Quartile at t+1"], maintain_order=True)
        .agg(
            pl.col("Prevalence").first(),
            pl.col("value").mean().round(2).cast(pl.String)
            + " ± "
            + pl.col("value").std(2).round(2).cast(pl.String),
        )
        .with_columns(pl.col("Prevalence").round(2))
        .with_columns(
            pl.when(pl.col("Metric").eq("AP"))
            .then(
                pl.col("value").add(" (" + pl.col("Prevalence").cast(pl.String)) + ")"
            )
            .otherwise(pl.col("value"))
        )
        .sort("Quartile at t+1")
        .collect()
        .pivot(on="Quartile at t+1", values="value", index=groups)
        .rename(RISK_GROUPS)
        .with_columns(
            pl.col("Metric").cast(pl.Enum(["AUROC", "AP"])),
            pl.col("Group").cast(GROUP_ORDER),
        )
        .sort("Factor model", "Predictor set", "Metric", "Group")
        .write_parquet("data/results/metrics/metric_summary.parquet")
    )


def quartile_metric_table(df: pl.DataFrame):
    return (
        df.filter(
            pl.col("Variable").eq("High-risk scenario")
            & pl.col("Predictor set").is_in(["Questionnaires", "CBCL scales"])
            & pl.col("Factor model").eq("Within-event")
        )
        .with_columns(
            pl.col("Group").cast(pl.Enum(["Conversion", "Persistence", "Agnostic"])),
            pl.col("Predictor set").cast(pl.Enum(["Questionnaires", "CBCL scales"])),
        )
        .drop("Factor model", "Variable")
        .sort("Predictor set", "Metric", pl.col("Group").cast(GROUP_ORDER))
        .rename({"Group": "High-risk scenario"})
    )


def demographic_metric_table(df: pl.DataFrame):
    return (
        df.filter(
            pl.col("Variable").ne("High-risk scenario")
            & pl.col("Predictor set").eq("Questionnaires")
            & pl.col("Factor model").eq("Within-event")
        )
        .drop("Factor model", "Predictor set")
        .sort("Metric", "Variable", pl.col("Group").cast(GROUP_ORDER))
    )


def make_shap_table(analysis: str, factor_model: str):
    columns = [
        "dataset",
        "table",
        "respondent",
        "question",
        "response",
    ]
    df = pl.read_csv(
        f"data/analyses/{factor_model}/{analysis}/results/shap_values.csv"
    ).rename({"Respondent": "respondent"})
    df = (
        df.group_by("variable")
        .agg(
            pl.col(columns).first(),
            pl.col("shap_value").sum().alias("shap_value"),
        )
        .sort(pl.col("shap_value").abs(), descending=True)
    ).select(["variable"] + columns + ["shap_value"])
    return df


def make_tables(cfg: Config):
    cross_tab = cross_tabulation()
    aggregate_metrics(
        analyses=cfg.experiment.analyses, factor_models=cfg.experiment.factor_models
    )
    df = pl.scan_parquet("data/results/metrics/metrics.parquet")
    groups = ["Factor model", "Predictor set", "Metric", "Variable", "Group"]
    make_metric_table(df=df, groups=groups)
    metric_table = pl.read_parquet("data/results/metrics/metric_summary.parquet")
    quartile_metrics = quartile_metric_table(df=metric_table)
    demographic_metrics = demographic_metric_table(df=metric_table)
    variable_metadata = pl.read_csv("data/raw/variable_metadata.csv")
    # aces = pl.read_excel("data/raw/ABCD_ACEs.xlsx")
    shap_table = make_shap_table(analysis="questions", factor_model="within_event")

    cross_tab.write_excel("data/tables/table_1.xlsx")
    quartile_metrics.write_excel("data/tables/table_2.xlsx")
    demographic_metrics.write_excel("data/tables/table_3.xlsx")

    variable_metadata.write_excel("data/supplement/tables/supplementary_table_1.xlsx")
    # aces.write_excel("data/supplement/tables/supplementary_table_2.xlsx")
    metric_table.write_excel("data/supplement/tables/supplementary_table_3.xlsx")
    shap_table.write_excel("data/supplement/tables/supplementary_table_4.xlsx")


if __name__ == "__main__":
    cfg = get_config(factor_model="within_event")
    make_tables(cfg)

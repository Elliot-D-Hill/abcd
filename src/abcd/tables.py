from collections.abc import Iterable
from itertools import product
from pathlib import Path

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


def aggregate_metrics(analyses: Iterable[tuple[str, str]], subanalysis: str):
    predictor_set_mapping = {
        "questions": "Participant Analysis"
        if subanalysis == "generalize"
        else "Questionnaires",
        "symptoms": "CBCL scales",
        "questions_symptoms": "Questionnaires, CBCL scales",
        "questions_mri": "Questionnaires, Theory-Driven MRI",
        "questions_mri_all": "Questionnaires, Data-Driven MRI",
        "questions_mri_symptoms": "Questionnaires, MRI, CBCL scales",
        "autoregressive": "Previous p-factors",
        "site": "Site Analysis",
        "propensity": "Propensity Score Weighting",
    }
    factor_model_mapping = {
        "within_event": "Within-event",
        "across_event": "Across-event",
    }
    for metric_type in ["metrics"]:  # "curves", "sens_spec",
        metrics = []
        for analysis, factor_model in analyses:
            path = f"data/analyses/{factor_model}/{analysis}/results/eval/{metric_type}.parquet"
            metric = (
                pl.read_parquet(path)
                .with_columns(
                    pl.lit(factor_model).alias("Factor model"),
                    pl.lit(analysis).alias("Predictor set"),
                )
                .with_columns(
                    pl.col("Predictor set").replace(predictor_set_mapping),
                    pl.col("Factor model").replace(factor_model_mapping),
                )
                .with_columns(
                    cs.numeric().shrink_dtype(), cs.string().cast(pl.Categorical)
                )
            )
            metrics.append(metric)
        df = pl.concat(metrics)
        result_path = f"data/results/{subanalysis}/{metric_type}.parquet"
        df.write_parquet(result_path)


def make_metric_table(lf: pl.LazyFrame):
    groups = ["Factor model", "Predictor set", "Metric", "Variable", "Group"]
    df = (
        lf.group_by(groups + ["Quartile at t+1"], maintain_order=True)
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
    )
    df = (
        df.pivot(on="Quartile at t+1", values="value", index=groups)
        .rename(RISK_GROUPS)
        .with_columns(
            pl.col("Metric").cast(pl.Enum(["AUROC", "AP"])),
            pl.col("Group").cast(GROUP_ORDER),
        )
        .sort("Factor model", "Predictor set", "Metric", "Group")
        .lazy()
    )
    return df


def quartile_metric_table(df: pl.LazyFrame):
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


def general_metric_table():
    df = pl.scan_parquet("data/results/generalize/metrics.parquet")
    df = make_metric_table(lf=df)
    # df = pl.scan_parquet("data/results/generalize/metric_summary.parquet")
    predictor_sets = ["Questionnaires", "Site splits", "Propensity scores"]
    df = (
        df.filter(
            pl.col("Variable").eq("High-risk scenario")
            & pl.col("Predictor set").is_in(predictor_sets)
            & pl.col("Factor model").eq("Within-event")
        )
        .with_columns(
            pl.col("Group").cast(pl.Enum(["Conversion", "Persistence", "Agnostic"])),
            pl.col("Predictor set").cast(pl.Enum(predictor_sets)),
        )
        .drop("Factor model", "Variable")
        .sort("Predictor set", "Metric", pl.col("Group").cast(GROUP_ORDER))
        .rename({"Group": "High-risk scenario"})
    )
    return df


def demographic_metric_table(df: pl.LazyFrame):
    return (
        df.filter(
            pl.col("Variable").ne("High-risk scenario")
            & pl.col("Predictor set").eq("Questionnaires")
            & pl.col("Factor model").eq("Within-event")
        )
        .drop("Factor model", "Predictor set")
        .sort("Metric", "Variable", pl.col("Group").cast(GROUP_ORDER))
    )


def make_shap_table(filepath: Path):
    columns = ["dataset", "table", "respondent", "question", "response"]
    df = pl.scan_parquet(filepath)
    df = (
        df.group_by("variable")
        .agg(
            pl.col(columns).first(),
            pl.col("shap_value").sum().alias("shap_value_sum"),
        )
        .sort(pl.col("shap_value_sum").abs(), descending=True)
    ).select(["variable"] + columns + ["shap_value_sum"])
    return df


def tuning_table():
    cfg = get_config("config.toml", analysis="questions", factor_model="within_event")
    df = pl.scan_parquet(cfg.filepaths.data.results.study)
    df = df.rename(lambda x: x.replace("params_", ""))
    df = df.rename({"value": "validation_loss"})
    df = df.drop("number", "datetime_start", "datetime_complete", "duration", "state")
    df = df.sort("validation_loss")
    df = df.with_columns(pl.col("method").replace_strict(cfg.tuner.methods))
    return df.collect()


def naive_metric_table(filepath: str):
    df = pl.scan_parquet(filepath)
    df = make_metric_table(lf=df)
    predictor_sets = ["previous_p_factor"]
    df = (
        df.filter(
            pl.col("Variable").eq("High-risk scenario")
            & pl.col("Predictor set").is_in(predictor_sets)
            & pl.col("Factor model").eq("Within-event")
        )
        .with_columns(
            pl.col("Group").cast(pl.Enum(["Conversion", "Persistence", "Agnostic"])),
            pl.col("Predictor set").cast(pl.Enum(predictor_sets)),
        )
        .drop("Factor model", "Variable")
        .sort("Predictor set", "Metric", pl.col("Group").cast(GROUP_ORDER))
        .rename({"Group": "High-risk scenario"})
    )
    return df


def make_tables(cfg: Config):
    cross_tab = cross_tabulation()
    analyses = product(cfg.experiment.analyses, cfg.experiment.factor_models)
    aggregate_metrics(analyses=analyses, subanalysis="all")
    generalize = [
        ("propensity", "within_event"),
        ("site", "within_event"),
        ("questions", "within_event"),
    ]
    aggregate_metrics(analyses=generalize, subanalysis="generalize")
    naive = [("previous_p_factor", "within_event")]
    aggregate_metrics(analyses=naive, subanalysis="naive")
    # df = naive_metric_table()
    # df = df.collect()
    # print(df)
    # df.write_parquet("data/supplement/temp.parquet")
    df = general_metric_table()
    df = df.collect()
    df.write_parquet("data/supplement/tables/supplementary_table_5.parquet")
    df = pl.scan_parquet("data/results/metrics/metrics.parquet")
    make_metric_table(lf=df)
    aggregate_metrics(
        analyses=[("questions", "within_event")],  # , ("symptoms", "within_event")
        subanalysis="new",
    )
    lf = pl.scan_parquet("data/results/new/metrics.parquet")
    metric_table = make_metric_table(lf)
    quartile_metrics = quartile_metric_table(df=metric_table).collect()
    print(quartile_metrics)
    quartile_metrics.write_excel("data/tables/table_x.xlsx")

    demographic_metrics = demographic_metric_table(df=metric_table)

    aces = pl.read_excel("data/raw/ABCD_ACEs.xlsx")

    cross_tab.write_excel("data/tables/table_1.xlsx")
    quartile_metrics.write_excel("data/tables/table_2.xlsx")
    demographic_metrics.collect().write_excel("data/tables/table_3.xlsx")

    variable_metadata = pl.read_parquet("data/processed/variables.parquet")
    variable_metadata.write_excel("data/supplement/tables/supplementary_table_1.xlsx")
    aces.write_excel("data/supplement/tables/supplementary_table_2.xlsx")
    metric_table.collect().write_excel(
        "data/supplement/tables/supplementary_table_3.xlsx"
    )

    cfg = get_config("config.toml", analysis="questions", factor_model="within_event")
    shap_table = make_shap_table(filepath=cfg.filepaths.data.results.shap.shap_values)
    tune_table = tuning_table()
    shap_table = shap_table.collect()
    shap_table.write_excel("data/supplement/tables/supplementary_table_4.xlsx")
    tune_table.write_excel("data/supplement/tables/supplementary_table_5.xlsx")

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns

from abcd.config import get_config
from abcd.constants import RISK_MAPPING

FORMAT = "pdf"
DPI = 300


def quartile_curves(
    read_path: Path | str, write_path: Path | str, predictor_set: list[str]
):
    df = pl.read_parquet(read_path)
    df = (
        df.filter(
            pl.col("Quartile at t+1").eq(4)
            & pl.col("Variable").eq("High-risk scenario")
            & pl.col("Predictor set").is_in(predictor_set)
            & pl.col("Factor model").eq("Within-event")
            & pl.col("y").ne(0)
        )
        .drop("Factor model", "Variable")
        .with_columns(
            pl.col("Metric").cast(pl.Enum(["ROC", "PR"])),
            pl.col("Group").cast(pl.Enum(["Conversion", "Persistence", "Agnostic"])),
            pl.col("Predictor set").cast(pl.Enum(predictor_set)),
        )
        .sort("Predictor set", "Metric", "Group", "y")
    )
    g = sns.relplot(
        data=df.to_pandas(),
        x="x",
        y="y",
        hue="Predictor set",
        row="Metric",
        col="Group",
        kind="line",
        errorbar=None,
        palette="deep",
        facet_kws={"sharex": False, "sharey": False},
    )
    g.set_titles("{col_name} {row_name} curve")
    font_size = 24
    labels = ["a", "b", "c", "d", "e", "f"]
    grouped_df = df.partition_by(["Metric", "Group"], maintain_order=True)
    for i, (ax, label, group) in enumerate(zip(g.axes.flat, labels, grouped_df)):
        if i == 3:
            ax.set_ylabel("Precision (positive predictive value)")
        if i == 0:
            ax.set_ylabel("True positive rate (sensitivity)")
        if i <= 2:
            ax.text(0.05, 0.95, label, fontsize=font_size)
            ax.set_xlabel("False positive rate (Type I error)")
            ax.plot([0, 1], [0, 1], linestyle="--", color="black")
        else:
            ax.text(0.95, 0.95, label, fontsize=font_size)
            ax.set_xlabel("Recall (sensitivity)")
            ax.axhline(
                group.filter(pl.col("y").ne(0))["y"].min(),
                linestyle="--",
                color="black",
            )
        ax.set_ylim(0.0, 1.05)
    plt.subplots_adjust(hspace=0.3)  # , wspace=0.4
    plt.savefig(write_path, format=FORMAT)


def abs_shap_sum(lf: pl.LazyFrame):
    df = lf.collect()
    order = (
        df.group_by("respondent", "dataset")
        .agg(pl.col("shap_value").sum().abs())
        .sort("shap_value", descending=True)["dataset"]
        .to_list()
    )
    dfs = []
    for _ in range(1000):
        resampled = df.sample(fraction=1, with_replacement=True)
        resampled = resampled.group_by(["dataset", "respondent"]).agg(
            pl.col("shap_value").sum().abs()
        )
        dfs.append(resampled)
    df = pl.concat(dfs)
    plt.figure(figsize=(10, 6))
    g = sns.pointplot(
        data=df,
        x="shap_value",
        y="dataset",
        hue="respondent",
        linestyles="none",
        order=order,
        errorbar="pi",
        hue_order=["Parent", "Youth"],
    )
    g.set(ylabel="Predictor category", xlabel="Absolute SHAP value sum")
    plt.legend(title="Respondent")
    g.yaxis.grid(True)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    sns.move_legend(g, "lower right")
    plt.savefig(f"data/figures/figure_3.{FORMAT}", format=FORMAT, bbox_inches="tight")


def analysis_comparison():
    df = pl.read_parquet("data/results/metrics/metrics.parquet")
    df = df.filter(
        pl.col("Variable").eq("High-risk scenario")
        & pl.col("Metric").eq("AUROC")
        & pl.col("Factor model").eq("Within-event")
    ).with_columns(
        pl.col("Quartile at t+1").replace_strict(RISK_MAPPING),
        pl.col("Group").cast(pl.Enum(["Conversion", "Persistence", "Agnostic"])),
    )
    g = sns.catplot(
        data=df.to_pandas(),
        x="Quartile at t+1",
        y="value",
        kind="bar",
        hue="Predictor set",
        col="Group",
        errorbar="pi",
    )
    g.set_titles("{col_name}")
    g.set(ylim=(0.5, 1.0))
    g.set_axis_labels("Risk", "AUROC")
    plt.savefig(
        f"data/supplement/figures/supplementary_figure_1.{FORMAT}",
        format=FORMAT,
        bbox_inches="tight",
    )


def shap_plot(
    lf: pl.LazyFrame,
    filepath: str,
    textwrap_width: int,
    y_axis_label: str,
    figsize: tuple[int, int],
):
    plt.figure(figsize=figsize)
    n_display = 20
    cbcl_names = {
        "Attention": "Attention Problems",
        "Somatic": "Somatic Complaints",
        "Aggressive": "Aggressive Behavior",
        "Rulebreak": "Rule-Breaking Behavior",
        "Thought": "Thought Problems",
        "Withdep": "Withdrawn/Depressed",
        "Social": "Social Problems",
        "Anxdep": "Anxious/Depressed",
    }
    cbcl_names = {
        k + " cbcl syndrome scale (t-score)": v for k, v in cbcl_names.items()
    }
    lf.rename({"Respondent": "respondent"})
    # df = pl.read_parquet(shap_path)
    lf = lf.with_columns(pl.col("question").replace(cbcl_names))
    lf = lf.with_columns(
        (pl.col("respondent") + ": " + pl.col("dataset")).alias("dataset"),
    )
    df = lf.collect()
    n_bootstraps = 100
    dfs = []
    for _ in range(n_bootstraps):
        resampled = (
            df.sample(fraction=1.0, with_replacement=True)
            .group_by("respondent", "question")
            .agg(pl.col("dataset").first(), pl.col("shap_value").sum().abs())
        )
        dfs.append(resampled)
    df = pl.concat(dfs)
    order = (
        df.group_by("question")
        .agg(pl.col("shap_value").sum().abs())
        .sort(by="shap_value", descending=True)
        .head(n_display)["question"]
        .to_list()
    )
    df = df.filter(pl.col("question").is_in(order))
    g = sns.pointplot(
        data=df.to_pandas(),
        x="shap_value",
        y="question",
        hue="dataset",
        errorbar="ci",
        linestyles="none",
        order=order,
    )
    g.legend(title="Dataset")
    g.set(xlabel="Absolute SHAP value sum", ylabel=y_axis_label)
    g.set_yticks(g.get_yticks())
    labels = [
        textwrap.fill(label.get_text(), textwrap_width) for label in g.get_yticklabels()
    ]
    g.set_yticklabels(labels)
    g.yaxis.grid(True)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{filepath}.{FORMAT}", format=FORMAT, dpi=DPI)


def p_factor_model_comparison():
    df = pl.read_parquet("data/results/metrics/metrics.parquet")
    df = (
        df.filter(
            pl.col("Predictor set").is_in(["Questionnaires", "CBCL scales"])
            & pl.col("Group").eq("Conversion")
            & pl.col("Metric").eq("AUROC")
        )
        .drop("Metric", "Group")
        .with_columns(
            pl.col("Quartile at t+1")
            .replace_strict(RISK_MAPPING)
            .cast(pl.Enum(["None", "Low", "Moderate", "High"]))
        )
        .sort("Quartile at t+1")
        .unique()
        .with_columns(cs.categorical().cast(pl.String))
    )
    g = sns.catplot(
        data=df.to_pandas(),
        x="Quartile at t+1",
        y="value",
        kind="bar",
        hue="Factor model",
        col="Predictor set",
        errorbar="pi",
    )
    g.set(ylim=(0.5, 1.0))
    g.set_axis_labels("Risk group", "AUROC")
    for ax in g.axes.flat:
        ax.axhline(0.5, color="black", linestyle="--")
    plt.savefig(
        f"data/supplement/figures/supplementary_figure_4.{FORMAT}",
        format=FORMAT,
        bbox_inches="tight",
        dpi=DPI,
    )


def shap_scatter(lf: pl.LazyFrame):
    sns.set_theme(context="paper", style="darkgrid", palette="deep", font_scale=2.5)
    plt.figure(figsize=(6, 8))
    # remove one outlier from Family mental health history
    lf = lf.filter(pl.col("feature_value") < 20)
    lf = lf.filter(pl.col("dataset") != "Follow-up event")
    df = lf.collect()
    order = (
        df.group_by("dataset")
        .agg(pl.col("shap_value").sum().abs())
        .sort("shap_value", descending=True)["dataset"]
        .to_list()
    )
    g = sns.lmplot(
        data=df,  # type: ignore
        x="feature_value",
        y="shap_value",
        hue="respondent",
        col="dataset",
        col_wrap=4,
        col_order=order,
        order=2,
        ci=95,
        facet_kws={"sharey": False, "sharex": False},
        scatter_kws={"alpha": 0.15, "rasterized": True},
        hue_order=["Parent", "Youth"],
    )
    ax = g.axes[1]
    handles, labels = ax.get_legend_handles_labels()
    g.legend.remove()  # type: ignore
    for handle in handles:
        handle.set_alpha(0.9)
    legend_data = dict(zip(labels, handles))
    g.add_legend(
        legend_data=legend_data,
        title="Respondent",
        markerscale=3,
        bbox_to_anchor=(0.9, 0.5),
        loc="center left",
    )
    g.set_titles("")
    g.set_axis_labels("Predictor value", "SHAP Value")
    for ax, col_name in zip(g.axes.flat, g.col_names):
        ax.set_xlabel(col_name)
    plt.savefig(
        f"data/supplement/figures/supplementary_figure_5.{FORMAT}",
        format=FORMAT,
        bbox_inches="tight",
        dpi=DPI,
    )


def plot():
    sns.set_theme(context="paper", style="darkgrid", palette="deep", font_scale=1.5)
    cfg = get_config(
        path="config.toml", factor_model="within_event", analysis="questions"
    )
    # quartile_curves("data/results/all/curves.parquet")
    # quartile_curves(
    #     read_path="data/results/metrics/curves.parquet",
    #     write_path=f"data/figures/figure_2.{FORMAT}",
    #     predictor_set=["CBCL scales", "Questionnaires"],
    # )
    # quartile_curves(
    #     read_path="data/results/generalize/curves.parquet",
    #     write_path=f"data/supplement/figures/supplementary_figure_6.{FORMAT}",
    #     predictor_set=[
    #         "Participant Analysis",
    #         "Site Analysis",
    #         "Propensity Score Weighting",
    #     ],
    # )
    # analysis_comparison()
    shap_values = pl.scan_parquet(cfg.filepaths.data.results.shap.shap_values)
    # abs_shap_sum(shap_values)
    # shap_scatter(shap_values)
    # cfg = get_config(
    #     path="config.toml", factor_model="within_event", analysis="symptoms"
    # )
    # symptom_shap_values = pl.scan_parquet(
    #     "data/analyses/within_event/symptoms/results/shap_values.csv"
    # )
    # shap_plot(
    #     lf=symptom_shap_values,
    #     filepath="data/supplement/figures/supplementary_figure_2",
    #     textwrap_width=75,
    #     y_axis_label="CBCL syndrome scale",
    #     figsize=(9, 5),
    # )
    shap_plot(
        lf=shap_values,
        filepath="data/supplement/figures/supplementary_figure_3",
        textwrap_width=80,
        y_axis_label="Question",
        figsize=(14, 8),
    )
    # p_factor_model_comparison()

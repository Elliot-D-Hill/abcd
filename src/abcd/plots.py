import textwrap

import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns
from toml import load

from abcd.cfg import Config

FORMAT = "pdf"


def quartile_curves():
    df = pl.read_parquet("data/results/metrics/curves.parquet")
    df = (
        df.filter(
            pl.col("Quartile at t+1").eq(4)
            & pl.col("Variable").eq("High-risk scenario")
            & pl.col("Predictor set").is_in(["CBCL scales", "Questionnaires"])
            & pl.col("Factor model").eq("Within-event")
            & pl.col("y").ne(0)
        )
        .drop("Factor model", "Variable")
        .with_columns(
            pl.col("Metric").cast(pl.Enum(["ROC", "PR"])),
            pl.col("Group").cast(pl.Enum(["Conversion", "Persistence", "Agnostic"])),
            pl.col("Predictor set").cast(pl.String),
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
    plt.savefig(f"data/figures/figure_1.{FORMAT}", format=FORMAT)


RISK_MAPPING = {1: "None", 2: "Low", 3: "Moderate", 4: "High"}


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
    filepath: str,
    analysis: str,
    factor_model: str,
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
    sns.set_theme(style="darkgrid", palette="deep", font_scale=1.6)
    df = pl.read_csv(
        f"data/analyses/{factor_model}/{analysis}/results/shap_values.csv"
    ).with_columns(
        (pl.col("Respondent") + ": " + pl.col("dataset")).alias("Dataset"),
        pl.col("question").replace(cbcl_names),
    )
    n_bootstraps = 100
    dfs = []
    for _ in range(n_bootstraps):
        resampled = (
            df.sample(fraction=1.0, with_replacement=True)
            .group_by("Respondent", "question")
            .agg(pl.col("Dataset").first(), pl.col("shap_value").sum())
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
        hue="Dataset",
        errorbar="ci",
        linestyles="none",
        order=order,
    )
    # sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    g.set(
        xlabel="SHAP value",
        ylabel=y_axis_label,
    )
    g.set_yticks(g.get_yticks())
    labels = [
        textwrap.fill(label.get_text(), textwrap_width) for label in g.get_yticklabels()
    ]
    g.set_yticklabels(labels)
    g.yaxis.grid(True)

    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{filepath}.{FORMAT}", format=FORMAT)


def grouped_shap_plot(
    filepath: str, analysis: str, factor_model: str, figsize: tuple[int, int]
):
    plt.figure(figsize=figsize)
    df = pl.read_csv(f"data/analyses/{factor_model}/{analysis}/results/shap_values.csv")
    n_bootstraps = 1000
    dfs = []
    for _ in range(n_bootstraps):
        resampled = (
            df.sample(fraction=1.0, with_replacement=True)
            .group_by("dataset", "Respondent")
            .agg(pl.col("shap_value").sum())
        )
        dfs.append(resampled)
    df = pl.concat(dfs)
    order = (
        df.group_by("dataset", "Respondent")
        .agg(pl.col("shap_value").sum().abs())
        .sort("shap_value", descending=True)["dataset"]
        .to_list()
    )
    g = sns.pointplot(
        data=df.to_pandas(),
        x="shap_value",
        y="dataset",
        hue="Respondent",
        linestyles="none",
        order=order,
        errorbar="pi",
    )
    g.set_yticks(g.get_yticks())
    labels = [textwrap.fill(label.get_text(), 28) for label in g.get_yticklabels()]
    g.set_yticklabels(labels)
    # g.yaxis.grid(True)
    sns.move_legend(g, "lower right")
    g.set(ylabel="Predictor category", xlabel="Summed SHAP value")
    g.yaxis.grid(True)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{filepath}.{FORMAT}", format=FORMAT)


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
    )


def plot(cfg):
    sns.set_theme(style="darkgrid", palette="deep", font_scale=2.0)
    sns.set_context("paper", font_scale=1.4)

    # quartile_curves()
    grouped_shap_plot(
        filepath="data/figures/small_figure_2",
        analysis="questions",
        factor_model="within_event",
        figsize=(7, 4),
    )
    # analysis_comparison()
    # shap_plot(
    #     filepath="data/supplement/figures/supplementary_figure_2",
    #     analysis="symptoms",
    #     factor_model="within_event",
    #     textwrap_width=75,
    #     y_axis_label="CBCL syndrome scale (t-score)",
    #     figsize=(16, 8),
    # )
    # shap_plot(
    #     filepath="data/supplement/figures/supplementary_figure_3",
    #     analysis="questions",
    #     factor_model="within_event",
    #     textwrap_width=75,
    #     y_axis_label="Question",
    #     figsize=(24, 16),
    # )
    # p_factor_model_comparison()


if __name__ == "__main__":
    cfg = Config(**load("cfg.toml"))
    plot(cfg)

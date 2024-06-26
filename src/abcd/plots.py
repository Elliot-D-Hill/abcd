import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
from torch import load as torch_load

# from shap import summary_plot
from matplotlib.patches import Patch
import textwrap

from abcd.config import Config


FORMAT = "png"


def predicted_vs_observed_plot():
    df = pl.read_csv("data/results/predicted_vs_observed.csv")
    print(df)
    df = df.rename(
        {
            "eventname": "Year",
            "race_ethnicity": "Race/Ethnicity",
            "demo_sex_v2_1": "Sex",
            "interview_age": "Age",
        }
    )
    df = (
        df.melt(
            id_vars=["y_pred", "y_true"],
            value_vars=["Year", "Race/Ethnicity", "Sex", "Age"],
        )
        .rename({"value": "Group", "variable": "Variable"})
        .with_columns(pl.col("Group").replace("9", " 9"))
        .sort("Group")
        .drop_nulls()
    )
    min_val = df.select(pl.min_horizontal(["y_true", "y_pred"]).min()).item()
    max_val = df.select(pl.max_horizontal(["y_true", "y_pred"]).max()).item()
    df = df.to_pandas()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    groups = df.groupby("Variable")
    palette = sns.color_palette("deep")
    for j, (ax, (name, group)) in enumerate(zip(axes.flat, groups)):
        labels = group["Group"].unique().tolist()
        for i, hue_category in enumerate(labels):
            hue_subset = group[group["Group"] == hue_category]
            sns.barplot(x="Group", y="r2", data=hue_subset, color=palette[i], ax=ax)
        handles = [Patch(facecolor=palette[i]) for i in range(len(labels))]
        ax.legend(
            handles=handles,
            labels=labels,
            title=name,
            loc="upper left",
        )
        if j in (0, 2):
            ax.set_ylabel("Predicted")
        if j in (2, 3):
            ax.set_xlabel("Observed")
        if j in (1, 3):
            ax.set_ylabel("")
        if j in (0, 1):
            ax.set_xlabel("")
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"data/plots/predicted_vs_observed.{FORMAT}", format=FORMAT)


def shap_plot(metadata):
    n_display = 20
    plt.figure(figsize=(16, 10))
    metadata = metadata.rename(
        {"column": "variable", "dataset": "Dataset"}
    ).with_columns((pl.col("Dataset") + " " + pl.col("respondent")).alias("Dataset"))
    df = pl.read_csv("data/results/shap_coefs.csv")
    df = df.join(other=metadata, on="variable", how="inner")
    # print(pl.col("Dataset").eq(pl.lit("Adverse childhood experiences")))
    f = pl.col("Dataset").eq(pl.lit("Adverse childhood experiences Parent"))
    print(df.filter(f))
    df = df.with_columns(  # FIXME move to variable.csv generation code
        pl.when(pl.col("variable").eq(pl.lit("total_core")))
        .then(pl.lit("Adverse childhood experiences"))
        .when(pl.col("variable").eq(pl.lit("eventname")))
        .then(pl.lit("Measurement year"))
        .otherwise(pl.col("question"))
        .alias("question")
    ).with_columns(
        pl.when(pl.col("variable").eq(pl.lit("eventname")))
        .then(pl.lit("Spatiotemporal"))
        .otherwise(pl.col("Dataset"))
        .alias("Dataset")
    )
    print(df)
    top_questions = (
        df.group_by("question")
        .agg(pl.col("value").abs().mean())
        .sort(by="value")
        .tail(n_display)["question"]
        .reverse()
        .to_list()
    )
    print(top_questions)
    df = (
        df.filter(pl.col("question").is_in(top_questions))
        .select(["question", "value", "Dataset"])
        .to_pandas()
    )
    g = sns.pointplot(
        data=df,
        x="value",
        y="question",
        hue="Dataset",
        errorbar=("sd", 2),
        linestyles="none",
        order=top_questions,
    )
    g.set(
        xlabel="SHAP value coefficient (impact of feature on model output)",
        ylabel="Feature description",
    )
    handles, labels = g.get_legend_handles_labels()
    sorted_labels, sorted_handles = zip(
        *sorted(zip(labels, handles), key=lambda t: t[0])
    )
    g.legend(sorted_handles, sorted_labels, loc="lower left", title="Dataset")
    g.autoscale(enable=True)
    g.set_yticks(g.get_yticks())
    labels = [textwrap.fill(label.get_text(), 75) for label in g.get_yticklabels()]
    g.set_yticklabels(labels)
    g.yaxis.grid(True)
    # ax2 = g.twinx() # TODO add response
    # ax2.set_yticks(g.get_yticks())
    # right_labels = [label for label in g.get_yticklabels()]
    # ax2.set_yticklabels(right_labels)
    # ax2.set_ylabel("Response")
    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"data/plots/shap_coefs.{FORMAT}", format=FORMAT)


def grouped_shap_plot(shap_values, X, column_mapping):
    plt.figure(figsize=(10, 8))
    df = pl.DataFrame(pd.DataFrame(shap_values, columns=X.columns))
    df = df.transpose(include_header=True, header_name="variable")
    df = (
        df.with_columns(pl.col("variable").replace(column_mapping).alias("dataset"))
        .group_by("dataset")
        .sum()
        .drop("variable")
    )
    columns = df.drop_in_place("dataset")
    # X = (  # FIXME this is kinda funky; refactor
    #     pl.DataFrame(X)
    #     .transpose(include_header=True, header_name="variable")
    #     .with_columns(pl.col("variable").replace(column_mapping).alias("dataset"))
    #     .group_by("dataset")
    #     .sum()
    #     .drop("variable")
    #     .transpose(column_names=columns)[1:]
    #     .to_pandas()
    #     .astype(float)
    # )
    df = df.transpose(column_names=columns).melt().with_columns(pl.col("value").abs())
    mean_values = (
        df.group_by("variable")
        .mean()
        .sort("value", descending=True)["variable"]
        .to_list()
    )
    g = sns.pointplot(
        data=df.to_pandas(),
        x="value",
        y="variable",
        linestyles="none",
        order=mean_values,
    )
    g.set(ylabel="Feature group", xlabel="Absolute summed SHAP values")
    g.yaxis.grid(True)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"data/plots/shap_grouped.{FORMAT}", format=FORMAT)


def shap_clustermap(shap_values, feature_names, column_mapping, color_mapping):
    plt.figure()
    sns.set_theme(style="white", palette="deep")
    sns.set_context("paper", font_scale=2.0)
    column_colors = {}
    color_mapping["ACEs"] = sns.color_palette("tab20")[-1]
    color_mapping["Spatiotemporal"] = sns.color_palette("tab20")[-2]
    for col, dataset in column_mapping.items():
        column_colors[col] = color_mapping[dataset]
    colors = [column_colors[col] for col in feature_names[1:] if col in column_colors]
    shap_df = pl.DataFrame(shap_values[:, 1:], schema=feature_names[1:]).to_pandas()
    shap_corr = shap_df.corr()
    g = sns.clustermap(
        shap_corr,
        row_colors=colors,
        yticklabels=False,
        xticklabels=False,
    )
    g.ax_col_dendrogram.set_visible(False)
    mask = np.triu(np.ones_like(shap_corr))
    values = g.ax_heatmap.collections[0].get_array().reshape(shap_corr.shape)  # type: ignore
    new_values = np.ma.array(values, mask=mask)
    g.ax_heatmap.collections[0].set_array(new_values)
    handles = [Patch(facecolor=color) for color in color_mapping.values()]
    plt.legend(
        handles,
        color_mapping.keys(),
        title="Dataset",
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(f"data/plots/shap_clustermap.{FORMAT}", format=FORMAT)


def plot(config: Config, dataloader):
    sns.set_theme(style="darkgrid", palette="deep")
    sns.set_context("paper", font_scale=1.5)
    # predicted_vs_observed_plot()
    names = [data["name"] for data in config.features.model_dump().values()]
    feature_names = (
        pl.read_csv("data/analytic/test.csv", n_rows=1)
        .drop(["src_subject_id", "p_factor"])
        .columns
    )
    test_dataloader = iter(dataloader)
    X, _ = next(test_dataloader)
    X = pd.DataFrame(X.view(-1, X.shape[2]), columns=feature_names)
    shap_values_list = torch_load("data/results/shap_values.pt")
    shap_values = np.mean(shap_values_list, axis=-1)
    shap_values = shap_values.reshape(-1, shap_values.shape[2])
    metadata = pl.read_csv("data/variables.csv")
    shap_plot(metadata=metadata)
    column_mapping = dict(zip(metadata["column"], metadata["dataset"]))
    grouped_shap_plot(
        shap_values=shap_values,
        X=X,
        column_mapping=column_mapping,
    )
    color_mapping = dict(zip(names, sns.color_palette("tab20")))
    shap_clustermap(
        shap_values=shap_values,
        feature_names=feature_names,
        column_mapping=column_mapping,
        color_mapping=color_mapping,
    )

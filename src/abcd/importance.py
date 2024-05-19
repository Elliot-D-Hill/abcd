from functools import partial
from shap import GradientExplainer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import polars as pl
from tqdm import tqdm
import pandas as pd
import numpy as np
from captum.attr import GradientShap, IntegratedGradients


def predict(x, model):
    output = model(x)
    return output.sum(dim=1)  # [:, -1]


def make_shap_values(model, X, background, columns):
    f = partial(predict, model=model.to("mps:0"))
    explainer = GradientShap(f)
    shap_values = explainer.attribute(
        X.to("mps:0"),
        background.to("mps:0"),
        # target=labels.squeeze(-1).to("mps:0"),  # .long().flatten(0, 1)
    ).sum(dim=1)
    # explainer = GradientExplainer(model.to("mps:0"), background.to("mps:0"))
    # shap_values = explainer.shap_values(X.to("mps:0"))
    # shap_values = np.sum(shap_values, axis=(1, 3))  # [:, :, -1]
    return pl.DataFrame(shap_values.cpu().numpy(), schema=columns)


def regress_shap_values(shap_values, X):
    coefs = {name: [] for name in X.columns}
    n_bootstraps = 100
    for _ in tqdm(range(n_bootstraps)):
        X_resampled, shap_resampled = resample(X, shap_values)
        for col in shap_values.columns:
            coef = (
                make_pipeline(StandardScaler(), LinearRegression())
                .fit(X_resampled[[col]], shap_resampled[[col]])
                .named_steps["linearregression"]
                .coef_[0, 0]
            )
            coefs[col].append(coef)
    df = pl.DataFrame(coefs).melt()
    df.write_csv("data/results/shap_coefs.csv")
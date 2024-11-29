from itertools import product

import polars as pl
from lightning import seed_everything
from sklearn import set_config
from tqdm import tqdm

from abcd.config import Analyses, get_config
from abcd.dataset import ABCDDataModule
from abcd.evaluate import evaluate_model
from abcd.importance import make_shap
from abcd.metadata import make_metadata
from abcd.model import make_model
from abcd.plots import plot
from abcd.tables import make_tables
from abcd.transform import get_dataset
from abcd.tune import tune


def make_experiment(cfg: Analyses):
    analyses = product(cfg.analyses, cfg.factor_models)
    return tqdm(analyses, total=len(cfg.analyses) * len(cfg.factor_models))


def main():
    pl.Config().set_tbl_cols(14)
    set_config(transform_output="polars")
    cfg = get_config(factor_model="within_event", analysis="metadata")
    if cfg.regenerate:
        make_metadata(cfg=cfg)
    seed_everything(cfg.random_seed)

    experiment = make_experiment(cfg=cfg.analyses)
    for analysis, factor_model in experiment:
        if not any(
            [
                cfg.evaluate,
                cfg.tune,
                cfg.importance,
            ]
        ):
            continue
        cfg = get_config(factor_model, analysis=analysis)
        splits = get_dataset(cfg=cfg)
        data_module = ABCDDataModule(splits=splits, cfg=cfg)
        input_dim = 1 if analysis == "autoregressive" else splits["train"].shape[-1] - 2
        if cfg.tune:
            tune(
                cfg=cfg,
                data_module=data_module,
                input_dim=input_dim,
                output_dim=cfg.preprocess.n_quantiles,
            )
        model = make_model(
            input_dim=input_dim, output_dim=cfg.preprocess.n_quantiles, cfg=cfg
        )
        if cfg.evaluate:
            evaluate_model(data_module=data_module, cfg=cfg, model=model)
        if cfg.importance:
            make_shap(
                cfg=cfg,
                model=model,
                data_module=data_module,
                analysis=analysis,
                factor_model=factor_model,
            )
    if cfg.tables:
        make_tables(cfg=cfg)
    if cfg.plot:
        plot(cfg=cfg)


if __name__ == "__main__":
    main()

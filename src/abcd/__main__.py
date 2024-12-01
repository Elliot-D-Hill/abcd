from itertools import product

import polars as pl
from lightning import seed_everything
from sklearn import set_config
from tqdm import tqdm

from abcd.config import Experiment, get_config
from abcd.dataset import ABCDDataModule
from abcd.evaluate import evaluate_model
from abcd.importance import make_shap
from abcd.metadata import make_metadata
from abcd.plots import plot
from abcd.process import get_dataset
from abcd.tables import make_tables
from abcd.tune import tune


def make_experiment(cfg: Experiment):
    analyses = product(cfg.analyses, cfg.factor_models)
    return tqdm(analyses, total=len(cfg.analyses) * len(cfg.factor_models))


def main():
    pl.enable_string_cache()
    pl.Config().set_tbl_cols(14)
    set_config(transform_output="polars")
    cfg = get_config(factor_model="within_event", analysis="metadata")
    seed_everything(cfg.random_seed)
    if cfg.regenerate:
        make_metadata(cfg=cfg)
    experiment = make_experiment(cfg=cfg.experiment)
    for analysis, factor_model in experiment:
        if not any(
            [
                cfg.evaluate,
                cfg.tune,
                cfg.importance,
            ]
        ):
            continue
        cfg = get_config(factor_model=factor_model, analysis=analysis)
        splits = get_dataset(cfg=cfg)
        data_module = ABCDDataModule(splits=splits, cfg=cfg)
        # -3 for split, sample id, and label
        input_dim = 1 if analysis == "autoregressive" else splits["train"].shape[-1] - 3
        cfg.model.input_dim = input_dim
        if cfg.tune:
            tune(cfg=cfg, data_module=data_module)
        if cfg.evaluate:
            evaluate_model(cfg=cfg, data_module=data_module)
        if cfg.importance:
            make_shap(cfg=cfg, data_module=data_module)
    if cfg.tables:
        make_tables(cfg=cfg)
    if cfg.plot:
        plot(cfg=cfg)


if __name__ == "__main__":
    main()

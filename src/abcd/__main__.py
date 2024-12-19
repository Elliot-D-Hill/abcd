from itertools import product

import polars as pl
import torch
from lightning import seed_everything
from sklearn import set_config
from tqdm import tqdm

from abcd.config import Experiment, get_config
from abcd.dataset import ABCDDataModule
from abcd.evaluate import evaluate_model
from abcd.importance import estimate_importance
from abcd.metadata import make_metadata
from abcd.plots import plot
from abcd.process import get_dataset
from abcd.tables import make_tables
from abcd.tune import tune_model


def make_experiment(cfg: Experiment):
    analyses = product(cfg.analyses, cfg.factor_models)
    return tqdm(analyses, total=len(cfg.analyses) * len(cfg.factor_models))


def main():
    # os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    pl.Config().set_tbl_cols(14)
    set_config(transform_output="polars")
    cfg = get_config(
        path="config.toml", factor_model="within_event", analysis="metadata"
    )
    seed_everything(cfg.random_seed)
    torch.set_float32_matmul_precision("medium")
    pl.set_random_seed(cfg.random_seed)
    make_metadata(cfg=cfg)
    experiment = make_experiment(cfg=cfg.experiment)
    print(experiment)
    for analysis, factor_model in experiment:
        if not any([cfg.regenerate, cfg.evaluate, cfg.tune, cfg.importance]):
            continue
        cfg = get_config(
            path="config.toml", factor_model=factor_model, analysis=analysis
        )
        splits = get_dataset(cfg=cfg)
        data_module = ABCDDataModule(splits=splits, cfg=cfg)
        cfg.model.input_dim = data_module.train[0][0].shape[-1]
        print("Input dimension:", cfg.model.input_dim)
        if cfg.tune:
            tune_model(cfg=cfg, data_module=data_module)
        if cfg.evaluate:
            evaluate_model(cfg=cfg, data_module=data_module)
        if cfg.importance:
            estimate_importance(cfg=cfg, data_module=data_module)
    if cfg.tables:
        make_tables(cfg=cfg)
    if cfg.plot:
        plot(cfg=cfg)


if __name__ == "__main__":
    main()

import polars as pl
import polars.selectors as cs
from nanuk.transform import assign_splits, impute, pipeline, standardize

from abcd.config import Config
from abcd.preprocess import generate_data


def get_brain_features(cfg: Config):
    brain_datasets = (
        "mri_y_dti_fa_fs_at",
        "mri_y_rsfmr_cor_gp_gp",
        "mri_y_tfmr_sst_csvcg_dsk",
        "mri_y_tfmr_mid_alrvn_dsk",
        "mri_y_tfmr_nback_2b_dsk",
    )
    return [
        column
        for name, features in cfg.features.model_dump().items()
        for column in features["columns"]
        if name in brain_datasets
    ]


def get_features(df: pl.DataFrame, cfg: Config):
    brain_features = get_brain_features(cfg)
    exclude = [cfg.join_on[0], "race_ethnicity", "interview_date"]
    match cfg.analysis:
        case "metadata":
            df = df.select(cs.exclude(exclude[:-2]))
        case "mri":
            df = df.select(cs.exclude(exclude))
        case "questions_mri":
            df = df.select(cs.exclude(exclude + cfg.features.mh_p_cbcl.columns))
        case "questions":
            df = df.select(
                cs.exclude(exclude + cfg.features.mh_p_cbcl.columns + brain_features)
            )
        case "questions_symptoms":
            df = df.select(cs.exclude(exclude + brain_features))
        case "symptoms" | "autoregressive":
            df = df.select(cfg.features.mh_p_cbcl.columns)
        case "questions_mri_symptoms":
            df = df.select(cs.exclude(exclude))
        case _:
            raise ValueError(f"Invalid analysis: {cfg.analysis}")
    return df.columns


def make_dataset(cfg: Config) -> pl.DataFrame:
    df = generate_data(cfg=cfg)
    df.sink_parquet(cfg.filepaths.data.raw.dataset)
    instance_id, event = cfg.join_on
    train_frac = cfg.preprocess.train_size
    val_test_frac = (1 - train_frac) / 2
    splits = {"train": train_frac, "val": val_test_frac, "test": val_test_frac}
    assignment = assign_splits(splits=splits, group=instance_id, name="split")
    df = df.with_columns(assignment)
    columns = cs.numeric().exclude(cfg.join_on)
    train = columns.filter(pl.col("split").eq("train"))
    imputation = impute(columns, method="median", train=train)
    scale = standardize(columns, method="zscore", train=train)
    transforms = [imputation, scale]
    df = pipeline(df, transforms, over=event)
    df.sink_parquet(cfg.filepaths.data.processed.dataset)
    return df.collect()


def get_dataset(cfg: Config) -> dict[str, pl.DataFrame]:
    if cfg.regenerate:
        df = make_dataset(cfg)
        splits = df.partition_by("split", as_dict=True)
    else:
        df = pl.read_csv(cfg.filepaths.data.processed.dataset)
        splits = df.partition_by("split", as_dict=True)
    return {str(names[0]): split for names, split in splits.items()}

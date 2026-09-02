import polars as pl
from sklearn.decomposition import FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from .config import Index


def make_quartile_pipeline(n_quartiles: int, random_seed: int) -> Pipeline:
    """Single-factor p-factor score of the CBCL scales, binned into quartiles."""
    return make_pipeline(
        StandardScaler(),
        SimpleImputer(strategy="mean"),
        FactorAnalysis(n_components=1, random_state=random_seed),
        KBinsDiscretizer(
            n_bins=n_quartiles,
            encode="ordinal",
            strategy="quantile",
            quantile_method="linear",
            random_state=random_seed,
        ),
    )


def within_event_quartile(
    event: pl.DataFrame,
    cbcl: tuple[str, ...],
    index: Index,
    n_quartiles: int,
    random_seed: int,
) -> pl.DataFrame:
    """Fit the factor model and quartile cutoffs on this event's training subjects only."""
    pipeline = make_quartile_pipeline(n_quartiles=n_quartiles, random_seed=random_seed)
    train = event.filter(pl.col(index.split).eq("train")).select(cbcl)
    pipeline.fit(train.to_numpy())
    quartile = pipeline.transform(event.select(cbcl).to_numpy())  # (N, 1)
    quartile = pl.Series(index.previous_label, quartile[:, 0]).cast(pl.Int64)
    return event.select(index.sample_id, index.event).with_columns(quartile)


def make_labels(
    dataset: pl.DataFrame,
    cbcl: tuple[str, ...],
    index: Index,
    n_quartiles: int,
    random_seed: int,
) -> pl.DataFrame:
    """Quartile at t and at the subject's next observed event; rows without a next event are dropped."""
    quartiles = [
        within_event_quartile(
            event,
            cbcl=cbcl,
            index=index,
            n_quartiles=n_quartiles,
            random_seed=random_seed,
        )
        for _, event in dataset.group_by(index.event)
    ]
    next_quartile = (
        pl.col(index.previous_label).shift(-1).over(index.sample_id).alias(index.label)
    )
    return (
        pl.concat(quartiles)
        .sort(index.sample_id, index.event)
        .with_columns(next_quartile)
        .drop_nulls(index.label)
    )

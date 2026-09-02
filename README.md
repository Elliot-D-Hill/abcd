# The Duke Predictive Model of Adolescent Mental Health (Duke-PMA)

This repository contains the code used to train and evaluate the Duke-PMA developed in [*Prediction of Mental Health Risk in Adolescents*](https://www.nature.com/articles/s41591-025-03560-7) (Hill et al. 2025). To train the model, access to data from the [Adolescent Brain Cognitive Development (ABCD) Study](https://abcdstudy.org/) is required.

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Elliot-D-Hill/abcd.git
cd abcd
uv sync
```

## Usage

Place the assembled ABCD dataset at `data/dataset.csv` (one row per subject and event, holding the CBCL syndrome scales, the questionnaire and demographic variables, and the imaging columns), then run:

```bash
uv run abcd
```

The pipeline:

1. Splits subjects 80/10/10 into train, validation, and test. A subject never appears in more than one split.
2. Builds the outcome. Within each event, a single-factor model of the eight CBCL syndrome scales is fit on the training subjects and its score is binned into quartiles. The label of a row is the quartile at the subject's next observed event.
3. Keeps the questionnaire and demographic features only (no CBCL scales, no imaging), z-scores and median-imputes them with training-split statistics, and writes `data/analytic/{train,val,test}.parquet`.
4. Trains the LSTM with the hyperparameters in `config.toml` (the values selected by the tuning study in the paper), keeps the checkpoint with the lowest validation loss, writes the test-set predictions to `data/results/predictions.parquet` (`output` holds the four class logits), and prints the test-set AUROC.

## Modifying program behavior

Most arguments that affect the program behavior are consolidated in the `config.toml` file. This allows for easy modification of program behavior by editing the values in that file without changing the source code. The `config.toml` file is parsed and validated with Pydantic (`src/abcd/config.py`) and can be accessed as a convenient Python object within the source code. The file `src/abcd/__main__.py` is the entry point for the package where all high-level control flow is defined.

## Using Artificial Intelligence for Longitudinal Mental Health Prediction in the ABCD Study

### Methods diagram

![figure_1](https://github.com/user-attachments/assets/4960e764-50c4-4584-84b8-02063b40512c)

### Model performance

![figure_2](https://github.com/user-attachments/assets/877ee549-70b5-4380-91ee-7df0926ed69d)

### Feature importance via aggregated SHAP values

![figure_3](https://github.com/user-attachments/assets/60a11863-b0ce-4099-93e9-31db42549481)

### SHAP value directionalities. Higher SHAP values indicate a greater predicted risk of psychopathology

![supplementary_figure_5](https://github.com/user-attachments/assets/82bc84cb-80f1-4fec-b919-aea35510d3ed)

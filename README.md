# The Duke Predictive Model of Adolescent Mental Health (Duke-PMA)

This repository contains the code used to train and evaluate the Duke-PMA developed in [*Prediction of Mental Health Risk in Adolescents*](https://www.nature.com/articles/s41591-025-03560-7) (Hill et al. 2025). To train the model, access to data from the [Adolescent Brain Cognitive Development (ABCD) Study](https://abcdstudy.org/) is required.

## Installation

```bash
git clone https://github.com/Elliot-D-Hill/abcd.git
```

Within the project directory, create and activate a virtual environment using the environment.yml file:

```bash
conda env create -f environment.yml
conda activate venv
```

Then install the package

```bash
pip install -e .
```

The argument `-e` can be omitted if you do not intend to modify the source code, but there's no harm in including it.

## Usage

Once installed, you can run the package with:

```bash
python -m abcd
```

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

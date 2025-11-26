# Using Artificial Intelligence for Longitudinal Mental Health Prediction in the ABCD Study

## Installation

In the project directory, create and activate a virtual environment using the environment.yml file:

```bash
conda env create -f environment.yml
conda activate venv
```

Then install the package

```bash
pip install -e .
```

The argument argument `-e` stands for 'editable' mode. This setting makes it so that changes made to the package are reflected in the code behavior in real-time so that you don't have to keep reinstalling the package when you modify it.

## Usage

Once installed, you can run the package with:

```bash
python -m abcd
```

## Project structure and configuration

Most arguments that affect program behavior are consolidated in `config.toml`. This allows for easy modification of program behavior without changing the source code by editing the values in that file. The `config.toml` file is parsed and validated with Pydantic and can be accessed as a Python object in the package. The file `__main__.py` is the entry point for the package where all high-level control flow is defined.

## Methods diagram

![figure_1](https://github.com/user-attachments/assets/4960e764-50c4-4584-84b8-02063b40512c)

## Model performance

![figure_2](https://github.com/user-attachments/assets/877ee549-70b5-4380-91ee-7df0926ed69d)

## Feature importance via aggregated SHAP values

![figure_3](https://github.com/user-attachments/assets/60a11863-b0ce-4099-93e9-31db42549481)

## SHAP value directionalities. Higher SHAP values indicate a greater predicted risk of psychopathology

![supplementary_figure_5](https://github.com/user-attachments/assets/82bc84cb-80f1-4fec-b919-aea35510d3ed)

# MachineLearningSamples

Regression exercises in Python (pandas, scikit-learn, matplotlib/seaborn) that I worked through while studying machine learning. Each exercise uses a public dataset and a short task description.

## Contents
| Folder | What |
|---|---|
| `RegressionProject1/` | `insurance.ipynb` — predict medical charges from `insurance.csv`: encoding of categorical features, linear vs degree-2 polynomial regression, train/test RMSE and R² comparison |
| `Sources/GradientDescent/` | `GradientDescent.py` — gradient descent from scratch on the Boston house-prices data (standardised `LSTAT` → `MEDV`) |
| `Sources/Polynomial Regression/01 … 08` | One script per dataset, each with `info.txt` (domain, dataset, learning goals): Auto MPG, Urban Traffic Volume, Wine Quality, Concrete Compressive Strength, Airfoil Self-Noise, Combined Cycle Power Plant, Abalone Age, Energy Efficiency |

## Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter openpyxl xlrd
jupyter notebook RegressionProject1/insurance.ipynb
python "Sources/GradientDescent/GradientDescent.py"
```
Scripts expect to be run from their own folder (they read the CSV next to them or in `dataset/`).

## Datasets
Datasets come from public sources (mostly the UCI Machine Learning Repository; the Boston and insurance CSVs are widely distributed teaching datasets). Check each dataset's own readme (`dataset/info.txt`, `*.names`) for its licence and citation before reuse. The `info.txt` task descriptions are in Persian.

## Known gaps
- `08 Energy Efficiency` reads `dataset/ENB2012_data.xlsx`, which is not in the repository (download the UCI *Energy efficiency* dataset into that folder).
- Two script names contain a double dot (`UrbanTraffic..py`, `wine_quality_pipeline..py`); they still run with `python`.

## Status
Learning exercises, not production models. Next: hold-out validation with cross-validation, regularisation (Ridge/Lasso) and a consistent project layout.

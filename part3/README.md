# Part 3 — Predicting Final-Exam Pass/Fail from Quiz Preparation

Binary classification: predict whether a student **passes or fails** the final
exam (`Final_Exam_Score >= 0.5`) using only features derived from the quizzes
they did as preparation. Three models are compared and all are logged to
Weights & Biases:

| Model | Why it's a candidate |
|-------|----------------------|
| **Logistic Regression** | Linear, fast, interpretable baseline. |
| **XGBoost** | Gradient-boosted trees — strong default for tabular data. |
| **MLP** (PyTorch Lightning) | Neural net; captures non-linearities, needs more data/tuning. |

All three train on the **same** train/test split and log identical metrics
(accuracy, precision, recall, F1, ROC-AUC) plus a confusion matrix and ROC
curve, so they are directly comparable in the wandb project.

## Structure

```
part3/
├── config.yaml          # data split, per-model hyperparameters, wandb project
├── data/                # raw anonymised semester CSVs + Processed_Data.csv
├── src/
│   ├── data.py          # load / clean / merge raw CSVs -> tidy dataframe
│   ├── features.py      # build_dataset(): leakage-free, scaled train/test split
│   ├── datamodule.py    # Lightning DataModule for the MLP
│   ├── metrics.py       # shared metrics + wandb logging (used by every model)
│   ├── models/
│   │   ├── mlp.py             # MLP classifier
│   │   └── sklearn_models.py  # LogReg + XGBoost factories
│   └── train.py         # trains all 3 models -> 3 wandb runs
└── notebooks/
    ├── 01_eda.ipynb
    ├── 02_feature_selection.ipynb
    └── 03_model_comparison.ipynb
```

## Running

From the `part3/` folder:

```bash
# Train all three models and log them to wandb
poetry run python -m src.train

# Run offline (no wandb account / no upload)
WANDB_MODE=offline poetry run python -m src.train   # PowerShell: $env:WANDB_MODE="offline"
```

Hyperparameters live in `config.yaml`. Open the `quiz_prediction` project in
wandb to compare the three runs side by side, or run
`notebooks/03_model_comparison.ipynb` for an in-notebook comparison table and
plots.

### Configuring the feature set and run names

`config.yaml` controls which columns the models train on and how the three wandb
runs are named:

```yaml
data:
  feature_set: "all"        # "all" = every non-leakage column; "selected" = the list below
  selected_features:        # used only when feature_set == "selected"
    - "mdm_Score/10.00_mean"
    - "statistics_Attempts"
    # ... (see notebooks/02_feature_selection.ipynb for how this was derived)

logging:
  run_names:                # name of each model's wandb run
    logreg: "LogisticRegression"
    xgboost: "XGBoost"
    mlp: "MLP"
```

Switch `feature_set` to `"selected"` to train every model on the compact,
interpretable subset. Selecting a leakage column is rejected automatically.

> **No data leakage:** `src/features.py` drops all exam-derived columns
> (`MDM`, `Stat`, `Formalise`, `Cost`, `Final_Exam_Score`, `passed`) and the
> identifiers, and the scaler is fit on the training split only.

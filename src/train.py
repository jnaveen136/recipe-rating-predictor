"""End-to-end training: split -> compare models -> tune -> stack -> submit.

Run from the repository root:
    python -m src.train            # full run (~30-40 min on 4 CPUs)
    python -m src.train --fast     # skip tuning, smaller stack (~10 min)
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from src.features import TARGET, add_features
from src.models import SEED, logreg, lightgbm, make_pipeline, model_zoo, stacking

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, IMAGES = ROOT / "data", ROOT / "results", ROOT / "image"


def evaluate(name, pipe, X_tr, y_tr, X_va, y_va, rows):
    start = time.time()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_va)
    row = {
        "model": name,
        "train_accuracy": round(accuracy_score(y_tr, pipe.predict(X_tr)), 4),
        "val_accuracy": round(accuracy_score(y_va, pred), 4),
        "val_macro_f1": round(f1_score(y_va, pred, average="macro"), 4),
        "fit_seconds": round(time.time() - start, 1),
    }
    rows.append(row)
    print(f"{name:28s} train={row['train_accuracy']:.4f} val={row['val_accuracy']:.4f} "
          f"macroF1={row['val_macro_f1']:.4f} ({row['fit_seconds']}s)", flush=True)
    return pipe, pred


def tune(name, estimator, space, X, y, n_iter):
    """Randomised search with stratified 3-fold CV on the training split only.
    Returns the best params; the caller rebuilds the model with them."""
    search = RandomizedSearchCV(
        make_pipeline(estimator),
        {f"model__{k}": v for k, v in space.items()},
        n_iter=n_iter, scoring="accuracy",
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=1, verbose=0,
    )
    search.fit(X, y)
    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print(f"[tune] {name}: CV accuracy={search.best_score_:.4f} params={best}", flush=True)
    return best


def plot_comparison(results: pd.DataFrame, path: Path):
    res = results.sort_values("val_accuracy")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(res["model"], res["val_accuracy"], color="#4C72B0")
    baseline = res.loc[res["model"].str.startswith("Baseline"), "val_accuracy"].iloc[0]
    ax.axvline(baseline, color="#C44E52", ls="--", label=f"Baseline (always predict 5) = {baseline:.3f}")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_xlim(0.70, max(res["val_accuracy"]) + 0.03)
    ax.set_xlabel("Validation accuracy (20% stratified hold-out)")
    ax.set_title("Recipe rating prediction: model comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(fast: bool):
    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    X = add_features(train.drop(columns=[TARGET]))
    y = train[TARGET]
    X_test = add_features(test)

    # Stratified split keeps the (very imbalanced) rating proportions equal.
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"train={X_tr.shape} val={X_va.shape} test={X_test.shape}")
    print("class balance:", y.value_counts(normalize=True).round(3).to_dict(), flush=True)

    rows = []
    for name, est in model_zoo().items():
        evaluate(name, make_pipeline(est), X_tr, y_tr, X_va, y_va, rows)

    lr_params, lgb_params = {"C": 3.0}, {}
    if not fast:
        lr_params = tune("Logistic Regression", logreg(), {"C": loguniform(0.3, 30)}, X_tr, y_tr, n_iter=8)
        lgb_params = tune("LightGBM", lightgbm(), {
            "num_leaves": randint(15, 63),
            "learning_rate": loguniform(0.02, 0.15),
            "n_estimators": randint(200, 600),
            "colsample_bytree": uniform(0.15, 0.45),
            "min_child_samples": randint(10, 40),
        }, X_tr, y_tr, n_iter=8)
        evaluate("Logistic Regression (tuned)", make_pipeline(logreg(**lr_params)), X_tr, y_tr, X_va, y_va, rows)
        evaluate("LightGBM (tuned)", make_pipeline(lightgbm(**lgb_params)), X_tr, y_tr, X_va, y_va, rows)

    stack, stack_pred = evaluate("Stacking (LR+LGBM+XGB+RF)",
                                 make_pipeline(stacking(lgb_params, C=lr_params["C"])),
                                 X_tr, y_tr, X_va, y_va, rows)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "model_comparison.csv", index=False)
    plot_comparison(results, IMAGES / "model_comparison.png")

    report = classification_report(y_va, stack_pred, digits=3, zero_division=0)
    (RESULTS / "stacking_classification_report.txt").write_text(report)
    print(report)
    ConfusionMatrixDisplay.from_predictions(y_va, stack_pred, cmap="Blues")
    plt.title("Stacking: validation confusion matrix")
    plt.tight_layout()
    plt.savefig(IMAGES / "confusion_matrix.png", dpi=150)
    plt.close()

    # Final model: refit the stack on ALL labelled rows, then predict test.
    final = make_pipeline(stacking(lgb_params, C=lr_params["C"])).fit(X, y)
    submission = pd.DataFrame({"ID": range(1, len(test) + 1), "Rating": final.predict(X_test)})
    submission.to_csv(RESULTS / "submission.csv", index=False)

    (RESULTS / "best_params.json").write_text(json.dumps(
        {"logistic_regression": lr_params, "lightgbm": lgb_params}, indent=2, default=float))
    print("Saved results/, image/model_comparison.png, image/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="skip hyperparameter tuning")
    main(parser.parse_args().fast)

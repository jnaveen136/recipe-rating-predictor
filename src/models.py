"""Model zoo. Each entry is a full sklearn Pipeline: preprocessing + estimator.

Wrapping preprocessing inside the pipeline means TF-IDF vocabularies and
scalers are fit only on the training fold during cross-validation, so no
information from validation rows leaks into training.
"""
import lightgbm as lgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import build_preprocessor

SEED = 42


def logreg(C: float = 3.0):
    return LogisticRegression(C=C, max_iter=3000, random_state=SEED)


def random_forest():
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=1, max_features="sqrt",
        n_jobs=-1, random_state=SEED,
    )


def lightgbm(**params):
    defaults = dict(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    defaults.update(params)
    return lgb.LGBMClassifier(**defaults)


def xgboost(**params):
    defaults = dict(
        n_estimators=400, learning_rate=0.1, max_depth=6,
        colsample_bytree=0.3, subsample=0.8, tree_method="hist",
        random_state=SEED, n_jobs=-1,
    )
    defaults.update(params)
    return XGBClassifier(**defaults)


def stacking(lgb_params=None, xgb_params=None, C: float = 3.0):
    """Level-0: diverse models. Level-1: logistic regression on their
    out-of-fold class probabilities (cv=5), so the meta-model never sees
    predictions a base model made on its own training rows."""
    base = [
        ("logreg", logreg(C)),
        ("lightgbm", lightgbm(**(lgb_params or {}))),
        ("xgboost", xgboost(**(xgb_params or {}))),
        ("random_forest", random_forest()),
    ]
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=3000, random_state=SEED),
        stack_method="predict_proba",
        cv=5,
        n_jobs=1,
    )


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def model_zoo():
    return {
        "Baseline (always 5)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": logreg(),
        "KNN": KNeighborsClassifier(n_neighbors=15, metric="cosine", n_jobs=-1),
        "Random Forest": random_forest(),
        "LightGBM": lightgbm(),
        "XGBoost": xgboost(),
    }

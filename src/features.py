"""Feature engineering and preprocessing for recipe rating prediction.

Everything that turns a raw CSV row into model input lives here, so the
training script and the notebook use exactly the same transformations.
"""
import html
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "Rating"
TEXT_COL = "Recipe_Review"

NUMERIC_COLS = [
    "UserReputation", "ReplyCount", "ThumbsUpCount", "ThumbsDownCount", "BestScore",
    "Year", "Month", "Day", "Hour", "Weekend",
    "WeightedVote", "ReputationXVotes",
    "ReviewLength", "ExclamationCount", "QuestionCount",
]
CATEGORICAL_COLS = ["RecipeName", "DayOfWeek", "TimeOfDay"]

# Columns that are pure identifiers: unique per row (CommentID) or a
# high-cardinality user key. Feeding them to a model only lets it memorise
# rows, so they are dropped. RecipeNumber/RecipeCode duplicate RecipeName.
DROPPED_COLS = ["ID", "CommentID", "UserID", "UserName", "RecipeNumber", "RecipeCode"]


def clean_text(text) -> str:
    """Decode HTML entities (&#39; -> ') and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def signed_log1p(x):
    """log(1+|x|) keeping the sign, so negative values (WeightedVote) stay valid."""
    return np.sign(x) * np.log1p(np.abs(x))


def time_of_day(hour: int) -> str:
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 18:
        return "Afternoon"
    if 18 <= hour < 21:
        return "Evening"
    return "Night"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered columns. Works on train and test alike (no target used)."""
    df = df.copy()
    ts = pd.to_datetime(df["CreationTimestamp"], unit="s")
    df["Year"] = ts.dt.year
    df["Month"] = ts.dt.month
    df["Day"] = ts.dt.day
    df["Hour"] = ts.dt.hour
    df["DayOfWeek"] = ts.dt.day_name()
    df["TimeOfDay"] = df["Hour"].apply(time_of_day)
    df["Weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)

    # Community-vote signals: net votes, and votes weighted by reputation.
    df["WeightedVote"] = df["ThumbsUpCount"] - df["ThumbsDownCount"]
    df["ReputationXVotes"] = df["UserReputation"] * (df["ThumbsUpCount"] + df["ThumbsDownCount"])

    # Missing reviews are kept (as empty text) instead of dropping the row.
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
    df["ReviewLength"] = df[TEXT_COL].str.split().str.len().fillna(0)
    df["ExclamationCount"] = df[TEXT_COL].str.count("!")
    df["QuestionCount"] = df[TEXT_COL].str.count(r"\?")
    return df.drop(columns=["CreationTimestamp"] + [c for c in DROPPED_COLS if c in df.columns])


def build_preprocessor(max_features: int = 20000) -> ColumnTransformer:
    """Numeric -> impute+log+scale, categorical -> one-hot, text -> TF-IDF.

    The output stays a sparse matrix, so ~20k text features fit in memory.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),  # tames heavy-tailed counts
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    text = TfidfVectorizer(
        ngram_range=(1, 2),    # unigrams + bigrams ("not good" != "good")
        min_df=2,              # ignore words seen only once
        max_features=max_features,
        sublinear_tf=True,     # 1 + log(tf) dampens repeated words
        strip_accents="unicode",
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_COLS),
            ("cat", categorical, CATEGORICAL_COLS),
            ("text", text, TEXT_COL),  # a single column name (not a list) -> 1-D input for TF-IDF
        ],
        sparse_threshold=1.0,
    )

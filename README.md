# Recipe Rating Prediction

Predict the star rating (0–5) a user gave a recipe from their **review text**, **community signals** (thumbs up/down, reputation, best score) and **posting time**. This was the IIT Madras BS *Machine Learning Practice* Kaggle project.

## Results

| Model | Val accuracy | Val macro-F1 |
|---|---|---|
| **Stacking (LR+LGBM+XGB+RF)** | 0.7984 | 0.384 |
| LightGBM (tuned) | 0.7903 | 0.328 |
| XGBoost | 0.7900 | 0.354 |
| LightGBM | 0.7889 | 0.356 |
| Logistic Regression | 0.7841 | 0.356 |
| Logistic Regression (tuned) | 0.7841 | 0.371 |
| Random Forest | 0.7753 | 0.239 |
| KNN | 0.7702 | 0.261 |
| Baseline (always 5) | 0.7606 | 0.144 |
| *Kaggle public leaderboard (stacking)* | *0.80004* | – |

Validation = 20% stratified hold-out. Tuning used 3-fold CV on the training split only.

<p align="center"><img src="image/model_comparison.png" width="650" alt="Model comparison"></p>

**Read the baseline first.** 76% of reviews are 5-star, so always predicting 5 already scores 0.761 accuracy. That's why macro-F1, which weights all six classes equally, is reported alongside accuracy. The models mainly improve on the rare 0–4 star classes. Rating 0 means the user commented *without* leaving a rating.

## Approach

```
raw CSV ──► add_features() ──► ColumnTransformer ───────────────────────► model
             time features      numeric: impute → signed log1p → scale
             vote features      categorical: one-hot (recipe, weekday, time of day)
             text statistics    text: TF-IDF (1–2 grams, 20k terms, sublinear tf)
```

- **Unstructured + structured data in one sklearn `Pipeline`.** Preprocessing is fit only on training folds, so there's no leakage into validation or CV. The matrix stays sparse the whole way (about 20k features).
- **Missing data** (empty reviews) is kept and imputed, not dropped.
- **Identifier columns are dropped**: `CommentID`, `UserID`, `UserName` and the numeric recipe codes. They only let a model memorise rows.
- **Models:** Logistic Regression, KNN, Random Forest (bagging), LightGBM and XGBoost (boosting), each tuned with `RandomizedSearchCV`. They're combined in a **stacking ensemble** (LR + LightGBM + XGBoost + RF → logistic-regression meta-model on 5-fold out-of-fold probabilities).

## Repository layout

```
data/                  train.csv, test.csv, sample.csv (submission format)
src/features.py        feature engineering + preprocessing ColumnTransformer
src/models.py          model definitions and the stacking ensemble
src/train.py           end-to-end: compare → tune → stack → submission
notebooks/             EDA and walkthrough (reads results produced by src/train.py)
results/               model_comparison.csv, classification report, best params, submission.csv
image/                 model comparison and confusion matrix plots
```

## Run it

```bash
pip install -r requirements.txt
python -m src.train          # full run with tuning (~30–40 min on 4 CPUs)
python -m src.train --fast   # no tuning (~10 min)
jupyter notebook notebooks/recipe_rating_prediction.ipynb
```

## Future work
- Fine-tune a transformer (e.g. DistilBERT) or use sentence embeddings for the review text.
- Use class weighting or threshold tuning if minority-class recall matters more than accuracy.
- Model "no rating" (class 0) as a separate detection step.

"""Re-run the thesis models (CountVectorizer + LogisticRegression, 10-fold KFold, same as
src/aspect_classification.py and src/sentiment_analysis.py) and save out-of-fold predictions,
so Jev and the thesis can be scored with the same code on the same rows.

    python baseline.py   ->  baseline_predictions.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from clean_text import CleanText  # noqa: E402  (the thesis preprocessing, unchanged)

from classify import load_reviews  # noqa: E402
from questions import ASPECTS  # noqa: E402

KEYS = list(ASPECTS.values())


def main():
    df = load_reviews()
    X = df["review"].reset_index(drop=True)
    kf = KFold(n_splits=10)  # thesis setting: no shuffle
    out = pd.DataFrame({"id": df["id"]})

    # Aspect classification: one-vs-rest LR on uni+bigrams, as in aspect_classification.py
    Y = df[[f"gold_mention_{k}" for k in KEYS]].to_numpy().astype(int)
    pred = np.zeros_like(Y)
    for tr, te in kf.split(X):
        model = Pipeline([
            ("preprocess", CleanText()),
            ("vect", CountVectorizer(ngram_range=(1, 2))),
            ("clf", OneVsRestClassifier(LogisticRegression())),
        ]).fit(X[tr], Y[tr])
        pred[te] = model.predict(X[te])
    for i, k in enumerate(KEYS):
        out[f"pred_mention_{k}"] = pred[:, i].astype(bool)

    # Sentiment: one LR per aspect, trained only on reviews labelled with that aspect,
    # as in sentiment_analysis.py (text cleaned up front, unigrams).
    clean = CleanText().fit_transform(X)
    for k in KEYS:
        gold = df[f"gold_sentiment_{k}"].reset_index(drop=True)
        rows = gold.dropna().index.to_numpy()
        Xk, yk = clean[rows].to_numpy(), gold[rows].to_numpy()
        preds = np.empty(len(rows), dtype=object)
        for tr, te in kf.split(Xk):
            model = Pipeline([("vect", CountVectorizer()), ("clf", LogisticRegression())]).fit(Xk[tr], yk[tr])
            preds[te] = model.predict(Xk[te])
        col = pd.Series([None] * len(df), dtype=object)
        col[rows] = preds
        out[f"pred_sentiment_{k}"] = col.to_numpy()

    out.to_csv(HERE / "baseline_predictions.csv", index=False)
    print(f"saved out-of-fold predictions for {len(out)} reviews -> baseline_predictions.csv")


if __name__ == "__main__":
    main()

"""
Model 8 – Classification with Embedding Features
--------------------------------------------------
Trains a Logistic Regression classifier on interaction vectors
(user_emb ∥ course_emb) to predict whether a user will complete a course.

Labels:
  rating 3.0 → 1 (completed)
  rating 2.0 → 0 (audited)

In this dataset ratings are only 2.0 and 3.0, but class_weight='balanced'
is used because the dataset is heavily imbalanced (222k completions vs
11k audits) — without it the classifier predicts class 1 for everything
and becomes useless as a discriminator.

Score = P(class 1) = probability the user will complete the course.

Cold-start fix for new users:
  Derive user embedding by averaging course embeddings of selected courses.
"""

import pandas as pd
import numpy as np
import os
import backend as backend_core

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

U_FEAT_COLS = [f"UFeature{i}" for i in range(16)]
C_FEAT_COLS = [f"CFeature{i}" for i in range(16)]

_clf_model     = None
_user_emb_df   = None
_course_emb_df = None


def _load_embeddings():
    global _user_emb_df, _course_emb_df
    if _user_emb_df is None:
        _user_emb_df = (
            pd.read_csv(os.path.join(DATA_DIR, "user_embeddings.csv"))
            .set_index("user")[U_FEAT_COLS]
        )
    if _course_emb_df is None:
        _course_emb_df = (
            pd.read_csv(os.path.join(DATA_DIR, "course_embeddings.csv"))
            .set_index("item")[C_FEAT_COLS]
        )


def _get_user_vector(user_id: int, enrolled_ids: list) -> np.ndarray:
    if user_id in _user_emb_df.index:
        return _user_emb_df.loc[user_id].values.astype(float)
    in_emb = [c for c in enrolled_ids if c in _course_emb_df.index]
    if in_emb:
        return _course_emb_df.loc[in_emb].values.mean(axis=0).astype(float)
    return _user_emb_df.values.mean(axis=0).astype(float)


def _build_training_data():
    _load_embeddings()
    ratings_df = backend_core.load_ratings()

    df = ratings_df.merge(_user_emb_df.reset_index(), on="user", how="inner")
    df = df.merge(_course_emb_df.reset_index(), on="item", how="inner")
    df.dropna(inplace=True)

    # Label: 3.0 = completed → 1,  2.0 = audited → 0
    # Any other rating value also maps to 0 (conservative)
    df["label"] = (df["rating"] == 3.0).astype(int)

    X = df[U_FEAT_COLS + C_FEAT_COLS].values.astype(float)
    y = df["label"].values.astype(int)
    return X, y


# ── Public API ────────────────────────────────────────────────────────────────

def train(params: dict = None):
    global _clf_model
    from sklearn.linear_model import LogisticRegression

    params = params or {}
    X, y   = _build_training_data()

    # class_weight='balanced' corrects for the heavy imbalance (95% class 1)
    # so the model actually learns to distinguish audited vs completed
    _clf_model = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced"
    )
    _clf_model.fit(X, y)


def predict(user_ids: list, params: dict = None) -> pd.DataFrame:
    """
    SCORE = P(completion) — probability the user will complete the course.
    """
    params      = params or {}
    top_courses = params.get("top_courses", 10)

    if _clf_model is None:
        train(params)

    _load_embeddings()
    ratings_df = backend_core.load_ratings()
    class_idx  = list(_clf_model.classes_).index(1) if 1 in _clf_model.classes_ else 0

    users, courses, scores = [], [], []

    for user_id in user_ids:
        enrolled_ids = ratings_df[ratings_df["user"] == user_id]["item"].tolist()
        enrolled_set = set(enrolled_ids)

        u_vec = _get_user_vector(user_id, enrolled_ids)

        raw_scores = {}
        for course_id, c_row in _course_emb_df.iterrows():
            if course_id in enrolled_set:
                continue
            interaction = np.concatenate([u_vec, c_row.values.astype(float)]).reshape(1, -1)
            prob = float(_clf_model.predict_proba(interaction)[0][class_idx])
            raw_scores[course_id] = prob

        if not raw_scores:
            continue

        top = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)[:top_courses]
        for course_id, score in top:
            users.append(user_id)
            courses.append(course_id)
            scores.append(round(score, 4))

    return pd.DataFrame({"USER": users, "COURSE_ID": courses, "SCORE": scores})

"""
Model 7 – Regression with Embedding Features
----------------------------------------------
IBM ML Capstone approach (lab_jupyter_cf_regression_w_embeddings):

Trains a Ridge regressor on interaction vectors (user_emb * course_emb,
element-wise) to predict the rating value. Recommends courses with the
highest predicted scores. The element-wise product (not concatenation)
is what makes the ranking differ per user — see _build_training_data.

Cold-start fix for new users (course selections from UI):
  New users have no row in user_embeddings.csv. Instead of the global
  mean, we derive their embedding by averaging the course embedding
  vectors of the courses they selected on the UI. This makes the course
  selection directly drive the recommendations.
"""

import pandas as pd
import numpy as np
import os
import backend as backend_core

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

U_FEAT_COLS = [f"UFeature{i}" for i in range(16)]
C_FEAT_COLS = [f"CFeature{i}" for i in range(16)]

_reg_model     = None
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
    """
    Return the user embedding vector.
    For new users, average the course embeddings of their selected courses.
    """
    if user_id in _user_emb_df.index:
        return _user_emb_df.loc[user_id].values.astype(float)

    # Cold-start: average selected course embeddings
    in_emb = [c for c in enrolled_ids if c in _course_emb_df.index]
    if in_emb:
        return _course_emb_df.loc[in_emb].values.mean(axis=0).astype(float)

    # Last resort: global mean user embedding
    return _user_emb_df.values.mean(axis=0).astype(float)


def _build_training_data():
    _load_embeddings()
    ratings_df = backend_core.load_ratings()

    df = ratings_df.merge(_user_emb_df.reset_index(), on="user", how="inner")
    df = df.merge(_course_emb_df.reset_index(), on="item", how="inner")
    df.dropna(inplace=True)

    # Element-wise product of the user and course embeddings (16-dim), NOT
    # concatenation. With concatenation the linear model score splits into
    # a user term + a course term that never interact, so for a fixed user
    # the course ranking is identical for everyone. The product lets each
    # user's dimensions scale the course's dimensions, so the ranking is
    # genuinely personalised (this is the interaction the lab intends).
    U = df[U_FEAT_COLS].values.astype(float)
    C = df[C_FEAT_COLS].values.astype(float)
    X = U * C
    y = df["rating"].values.astype(float)
    return X, y


# ── Public API ────────────────────────────────────────────────────────────────

def train(params: dict = None):
    global _reg_model
    from sklearn.linear_model import Ridge

    params = params or {}
    X, y   = _build_training_data()
    _reg_model = Ridge(alpha=1.0)
    _reg_model.fit(X, y)


def predict(user_ids: list, params: dict = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    user_ids    : list of int
    params      : dict, optional
        top_courses : int   Max recommendations per user. Default 10.
    """
    params      = params or {}
    top_courses = params.get("top_courses", 10)

    if _reg_model is None:
        train(params)

    _load_embeddings()
    ratings_df = backend_core.load_ratings()
    users, courses, scores = [], [], []

    for user_id in user_ids:
        enrolled_ids = ratings_df[ratings_df["user"] == user_id]["item"].tolist()
        enrolled_set = set(enrolled_ids)

        u_vec = _get_user_vector(user_id, enrolled_ids)

        raw_scores = {}
        for course_id, c_row in _course_emb_df.iterrows():
            if course_id in enrolled_set:
                continue
            interaction = (u_vec * c_row.values.astype(float)).reshape(1, -1)
            raw_scores[course_id] = float(_reg_model.predict(interaction)[0])

        if not raw_scores:
            continue

        top = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)[:top_courses]

        # SCORE is the raw predicted rating (ratings are 2.0–3.0 in this
        # dataset). Rescaling within the top-N would force the last
        # recommendation to display as 0.0, which reads as "no relevance".
        for course_id, score in top:
            users.append(user_id)
            courses.append(course_id)
            scores.append(round(score, 4))

    return pd.DataFrame({"USER": users, "COURSE_ID": courses, "SCORE": scores})

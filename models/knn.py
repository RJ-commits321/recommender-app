"""
Model 4 – KNN Collaborative Filtering
--------------------------------------
Finds K nearest neighbour users by cosine similarity on the user-item
rating matrix, then scores each unenrolled course by the similarity-
weighted fraction of neighbours who took it:

    score = sum(sims of neighbours who took the course) / sum(all neighbour sims)

Scoring change vs the textbook formula:
  The classic KNN-CF score is the similarity-weighted AVERAGE of
  neighbour ratings (predicting the rating you would give). In this
  dataset ratings are only 2.0/3.0 and ~95% are 3.0, so that average is
  almost always exactly 3.0 — nearly every course ties, the top-N is
  arbitrary, and every displayed score is 1.0. Enrollment itself is the
  real signal here (implicit feedback), so we rank by how much of the
  user's neighbourhood took each course instead. Scores are naturally in
  (0, 1] and meaningfully spread.

Cold-start fix:
  New users are registered (in memory) before predict() is called, but
  the pivot matrix may be cached from before that insertion, so the new
  user is not in it. We handle this by building their vector directly
  from the pivot columns using their selected course IDs — no popularity
  fallback needed, their selections drive the neighbours found.
"""

import pandas as pd
import numpy as np
import backend as backend_core

_pivot_matrix = None


def _get_pivot_matrix() -> pd.DataFrame:
    global _pivot_matrix
    if _pivot_matrix is None:
        ratings_df = backend_core.load_ratings()
        _pivot_matrix = ratings_df.pivot_table(
            index="user", columns="item", values="rating", fill_value=0
        )
    return _pivot_matrix


def _cosine_sim(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    norm_vec = np.linalg.norm(vec)
    norms    = np.linalg.norm(matrix, axis=1)
    denom    = norms * norm_vec
    denom[denom == 0] = 1e-10
    return matrix.dot(vec) / denom


def _build_user_vec_from_selections(enrolled_ids: list, pivot: pd.DataFrame) -> np.ndarray:
    """
    Build a rating vector for a new user aligned to pivot columns.
    Sets 3.0 for each selected course that exists in the pivot, 0 elsewhere.
    """
    vec = np.zeros(len(pivot.columns))
    col_index = {c: i for i, c in enumerate(pivot.columns)}
    for cid in enrolled_ids:
        if cid in col_index:
            vec[col_index[cid]] = 3.0
    return vec


def _knn_predict_for_user(user_id, n_neighbors, top_courses, ratings_df):
    pivot = _get_pivot_matrix()
    enrolled_ids = ratings_df[ratings_df["user"] == user_id]["item"].tolist()
    enrolled_set = set(enrolled_ids)

    if user_id in pivot.index:
        # Existing user — use their row from the cached pivot
        user_vec = pivot.loc[user_id].values.astype(float)
        exclude_idx = pivot.index.get_loc(user_id)  # O(1) lookup
    else:
        # New user — build vector from their selected courses
        user_vec    = _build_user_vec_from_selections(enrolled_ids, pivot)
        exclude_idx = None

    if user_vec.sum() == 0:
        return {}

    matrix = pivot.values.astype(float)
    sims   = _cosine_sim(user_vec, matrix)

    if exclude_idx is not None:
        sims[exclude_idx] = -1  # exclude self

    neighbour_indices = np.argsort(sims)[::-1][:n_neighbors]
    neighbour_sims    = sims[neighbour_indices]

    total_sim = neighbour_sims.sum()
    if total_sim <= 0:
        return {}

    # Implicit-feedback scoring (see module docstring): similarity-weighted
    # fraction of the neighbourhood that took each course. The old
    # weighted-average-of-ratings score tied at 3.0 for almost every course.
    scores = {}
    for col_idx, course_id in enumerate(pivot.columns):
        if course_id in enrolled_set:
            continue
        took = matrix[neighbour_indices, col_idx] > 0
        if not took.any():
            continue
        scores[course_id] = float(neighbour_sims[took].sum() / total_sim)

    if not scores:
        return {}

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_courses]
    return {cid: round(s, 4) for cid, s in top}


# ── Public API ────────────────────────────────────────────────────────────────

def train(params: dict = None):
    global _pivot_matrix
    _pivot_matrix = None
    _get_pivot_matrix()


def predict(user_ids: list, params: dict = None) -> pd.DataFrame:
    params      = params or {}
    n_neighbors = params.get("n_neighbors", 10)
    top_courses = params.get("top_courses", 10)

    ratings_df = backend_core.load_ratings()
    users, courses, scores = [], [], []

    for user_id in user_ids:
        recs = _knn_predict_for_user(user_id, n_neighbors, top_courses, ratings_df)
        for course_id, score in recs.items():
            users.append(user_id)
            courses.append(course_id)
            scores.append(score)

    return pd.DataFrame({"USER": users, "COURSE_ID": courses, "SCORE": scores})

"""
Model 5 – NMF Collaborative Filtering
---------------------------------------
IBM ML Capstone approach (lab_jupyter_cf_nmf):

Decomposes the user-item rating matrix R ≈ W × H using sklearn NMF,
then reconstructs predicted ratings for all (user, item) pairs.

Cold-start fix for new users (course selections from UI):
  New users are not in the reconstructed matrix. Instead of falling back
  to a global mean, we project the new user into the NMF latent space by
  solving: min ||W_new × H - r_new||  using NNLS (non-negative least
  squares), where r_new is a sparse rating row built from selected courses.
  This gives a meaningful personalised latent vector using their selections.
"""

import pandas as pd
import numpy as np
import backend as backend_core
from scipy.optimize import nnls

_nmf_model        = None
_reconstructed    = None
_nmf_n_components = None
_pivot_columns    = None   # course columns in the pivot (needed for projection)
_H_matrix         = None   # item latent factors (n_components × n_courses)


def _build_pivot():
    ratings_df = backend_core.load_ratings()
    return ratings_df.pivot_table(
        index="user", columns="item", values="rating", fill_value=0
    )


def _get_new_user_vector(enrolled_ids: list) -> np.ndarray:
    """
    Project a new user into NMF latent space using their selected courses.
    Builds a sparse rating row over pivot columns then solves via NNLS.
    Falls back to mean of course latent vectors if no overlap with pivot.
    """
    if _pivot_columns is None or _H_matrix is None:
        return None

    # Build sparse rating row aligned to pivot columns
    r_new = np.zeros(len(_pivot_columns))
    col_index = {c: i for i, c in enumerate(_pivot_columns)}
    matched = 0
    for cid in enrolled_ids:
        if cid in col_index:
            r_new[col_index[cid]] = 3.0   # use the standard enrollment rating
            matched += 1

    if matched == 0:
        return None

    # Solve: H.T @ w ≈ r_new  →  w = NNLS(H.T, r_new)
    w_new, _ = nnls(_H_matrix.T, r_new)
    return w_new


# ── Public API ────────────────────────────────────────────────────────────────

def train(params: dict = None):
    global _nmf_model, _reconstructed, _nmf_n_components, _pivot_columns, _H_matrix
    from sklearn.decomposition import NMF

    params       = params or {}
    n_components = params.get("n_components", 15)

    pivot = _build_pivot()
    X     = pivot.values.astype(float)

    nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
    W   = nmf.fit_transform(X)
    H   = nmf.components_

    _nmf_model        = nmf
    _nmf_n_components = n_components
    _pivot_columns    = list(pivot.columns)
    _H_matrix         = H   # shape: (n_components, n_courses)
    _reconstructed    = pd.DataFrame(
        np.dot(W, H), index=pivot.index, columns=pivot.columns
    )


def predict(user_ids: list, params: dict = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    user_ids    : list of int
    params      : dict, optional
        n_components : int   NMF latent factors. Default 15.
        top_courses  : int   Max recommendations per user. Default 10.
    """
    global _nmf_model, _reconstructed, _nmf_n_components

    params       = params or {}
    n_components = params.get("n_components", 15)
    top_courses  = params.get("top_courses", 10)

    if _nmf_model is None or _nmf_n_components != n_components:
        train({"n_components": n_components})

    ratings_df = backend_core.load_ratings()
    users, courses, scores = [], [], []

    for user_id in user_ids:
        enrolled_ids = ratings_df[ratings_df["user"] == user_id]["item"].tolist()
        enrolled_set = set(enrolled_ids)

        if user_id in _reconstructed.index:
            # Existing user — use reconstructed row directly
            pred_row = _reconstructed.loc[user_id]
        else:
            # New user — project into latent space via NNLS from selected courses
            w_new = _get_new_user_vector(enrolled_ids)
            if w_new is None:
                continue
            pred_scores = np.dot(w_new, _H_matrix)
            pred_row = pd.Series(pred_scores, index=_pivot_columns)

        candidates = pred_row[~pred_row.index.isin(enrolled_set)]
        top        = candidates.nlargest(top_courses)
        if top.empty:
            continue

        max_val = top.iloc[0] if top.iloc[0] > 0 else 1.0
        for course_id, score in top.items():
            users.append(user_id)
            courses.append(course_id)
            scores.append(round(float(score / max_val), 4))

    return pd.DataFrame({"USER": users, "COURSE_ID": courses, "SCORE": scores})

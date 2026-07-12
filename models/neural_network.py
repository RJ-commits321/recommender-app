"""
Model 6 – Neural Network Collaborative Filtering (PyTorch, GMF-style)
------------------------------------------------------------------------
Generalized Matrix Factorization (GMF): the "neural network" here learns
user and course embeddings end-to-end via gradient descent, and the
predicted rating is the dot product of those two embeddings — same
scoring function used at inference, so training and serving are
consistent (this fixes an earlier version that trained extra dense
layers but served with a plain dot product, ignoring what those
layers learned).

Architecture:
  user_id   ──► Embedding(n_users, dim) ──┐
                                            ├──► dot product ──► predicted rating
  course_id ──► Embedding(n_items, dim) ──┘

Loss: MSE between dot(user_emb, item_emb) and actual rating.

Cold-start fix for new users:
  New users have no trained embedding row. We derive their embedding by
  averaging the course embeddings of the courses they selected on the UI.

If PyTorch is not available at runtime, falls back to the pre-computed
embeddings in user_embeddings.csv / course_embeddings.csv (themselves
produced by an equivalent embedding model) and still applies the same
course-average cold-start fix.
"""

import pandas as pd
import numpy as np
import os
import backend as backend_core

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

U_FEAT_COLS = [f"UFeature{i}" for i in range(16)]
C_FEAT_COLS = [f"CFeature{i}" for i in range(16)]

_user_embeddings   = None
_course_embeddings = None
_embeddings_are_calibrated = False   # True once trained fresh via PyTorch (already rating-scale)


def _load_pretrained_embeddings():
    global _user_embeddings, _course_embeddings
    if _user_embeddings is None:
        _user_embeddings = (
            pd.read_csv(os.path.join(DATA_DIR, "user_embeddings.csv"))
            .set_index("user")[U_FEAT_COLS]
        )
    if _course_embeddings is None:
        _course_embeddings = (
            pd.read_csv(os.path.join(DATA_DIR, "course_embeddings.csv"))
            .set_index("item")[C_FEAT_COLS]
        )


def _calibrate_scores(raw_scores: dict, ratings_df: pd.DataFrame) -> dict:
    """
    Pre-computed embeddings from an external source have dot products on a
    totally different scale than actual ratings (e.g. ~0.01 vs ~2-3).
    Linearly map the user's candidate scores onto the rating scale using
    the actual min/max of those dot products — no hard-coded range, so no
    clipping and no ties at the top. The map is monotone, so ranking is
    unchanged.
    """
    r_min, r_max = ratings_df["rating"].min(), ratings_df["rating"].max()
    lo, hi = min(raw_scores.values()), max(raw_scores.values())
    rng = hi - lo if hi > lo else 1.0
    return {
        cid: r_min + (s - lo) / rng * (r_max - r_min)
        for cid, s in raw_scores.items()
    }


def _get_user_vector(user_id: int, enrolled_ids: list) -> np.ndarray:
    if _user_embeddings is not None and user_id in _user_embeddings.index:
        return _user_embeddings.loc[user_id].values.astype(float)
    if _course_embeddings is not None and enrolled_ids:
        in_emb = [c for c in enrolled_ids if c in _course_embeddings.index]
        if in_emb:
            return _course_embeddings.loc[in_emb].values.mean(axis=0).astype(float)
    if _user_embeddings is not None:
        return _user_embeddings.values.mean(axis=0).astype(float)
    return None


def _train_pytorch(ratings_df, params):
    """Train a GMF (pure embedding dot-product) model in PyTorch."""
    global _user_embeddings, _course_embeddings, _embeddings_are_calibrated
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    embedding_dim = params.get("embedding_dim", 16)
    epochs        = params.get("epochs", 10)
    batch_size    = 512

    users = ratings_df["user"].unique()
    items = ratings_df["item"].unique()
    u2idx = {u: i for i, u in enumerate(users)}
    i2idx = {it: i for i, it in enumerate(items)}

    X_u = torch.tensor(ratings_df["user"].map(u2idx).values, dtype=torch.long)
    X_i = torch.tensor(ratings_df["item"].map(i2idx).values, dtype=torch.long)
    y   = torch.tensor(ratings_df["rating"].values, dtype=torch.float32)

    dataset    = TensorDataset(X_u, X_i, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    class GMF(nn.Module):
        """Generalized Matrix Factorization: score = dot(user_emb, item_emb)."""
        def __init__(self, n_users, n_items, emb_dim):
            super().__init__()
            self.user_emb = nn.Embedding(n_users, emb_dim)
            self.item_emb = nn.Embedding(n_items, emb_dim)
            # Small init so dot products start near 0, helps convergence
            nn.init.normal_(self.user_emb.weight, std=0.05)
            nn.init.normal_(self.item_emb.weight, std=0.05)

        def forward(self, u_idx, i_idx):
            u = self.user_emb(u_idx)
            i = self.item_emb(i_idx)
            return (u * i).sum(dim=1)   # dot product per row

    model     = GMF(len(users), len(items), embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for u_batch, i_batch, y_batch in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(u_batch, i_batch), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        u_weights = model.user_emb.weight.numpy()
        i_weights = model.item_emb.weight.numpy()

    u_cols = [f"UFeature{j}" for j in range(embedding_dim)]
    c_cols = [f"CFeature{j}" for j in range(embedding_dim)]

    _user_embeddings   = pd.DataFrame(u_weights, index=users, columns=u_cols)
    _course_embeddings = pd.DataFrame(i_weights, index=items, columns=c_cols)
    _user_embeddings.index.name   = "user"
    _course_embeddings.index.name = "item"
    # Fresh PyTorch training already predicts directly on the rating scale
    # (loss was MSE against actual ratings), so no further calibration needed.
    _embeddings_are_calibrated = True


def train(params: dict = None):
    """Train via PyTorch GMF if available, else load pre-computed embeddings."""
    global _embeddings_are_calibrated
    params = params or {}
    try:
        import torch  # noqa
        ratings_df = backend_core.load_ratings()
        _train_pytorch(ratings_df, params)
    except ImportError:
        _load_pretrained_embeddings()
        _embeddings_are_calibrated = False


def predict(user_ids: list, params: dict = None) -> pd.DataFrame:
    """
    SCORE = dot(user_embedding, course_embedding).

    If embeddings came from fresh PyTorch training, this dot product was
    directly optimised against the rating scale (2.0–3.0) and is used as-is.

    If embeddings came from the pre-computed CSV fallback (no PyTorch
    available), their dot products are on an arbitrary external scale —
    we linearly rescale them onto the rating scale using the min/max of
    the user's own candidate scores, so SCORE is comparable to actual
    ratings for both display and RMSE/MAE evaluation.
    """
    params      = params or {}
    top_courses = params.get("top_courses", 10)

    if _user_embeddings is None or _course_embeddings is None:
        _load_pretrained_embeddings()

    ratings_df = backend_core.load_ratings()
    users, courses, scores = [], [], []

    for user_id in user_ids:
        enrolled_ids = ratings_df[ratings_df["user"] == user_id]["item"].tolist()
        enrolled_set = set(enrolled_ids)

        u_vec = _get_user_vector(user_id, enrolled_ids)
        if u_vec is None:
            continue

        raw_scores = {}
        for course_id, c_row in _course_embeddings.iterrows():
            if course_id in enrolled_set:
                continue
            raw_scores[course_id] = float(np.dot(u_vec, c_row.values.astype(float)))

        if not raw_scores:
            continue

        if not _embeddings_are_calibrated:
            raw_scores = _calibrate_scores(raw_scores, ratings_df)

        top = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)[:top_courses]
        for course_id, score in top:
            users.append(user_id)
            courses.append(course_id)
            scores.append(round(score, 4))

    return pd.DataFrame({"USER": users, "COURSE_ID": courses, "SCORE": scores})

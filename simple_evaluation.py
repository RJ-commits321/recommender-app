"""
simple_evaluation.py
======================================================================
One row per model, ONE evaluation pass, independent of whatever model
the user currently has selected in the sidebar. Computed on demand
(e.g. via a button) — not on every app rerun.
TO DISABLE: comment out the call site in app.py (search for
"import simple_evaluation"). This file can be left in place untouched.
──────────────────────────────────────────────────────────────────────
METRIC PLAN
──────────────────────────────────────────────────────────────────────
Course Similarity   → Avg Similarity Score   (no hold-out)
User Profile        → Avg Genre Match Score  (no hold-out)
Clustering          → Hit Rate@K + Silhouette Score   (shared hold-out)
Clustering w/ PCA   → Hit Rate@K + Silhouette Score   (shared hold-out)
KNN                 → Hit Rate@K             (shared hold-out)
NMF                 → RMSE + MAE             (shared hold-out)
Neural Network      → RMSE + MAE             (shared hold-out)
Regression           → RMSE + MAE             (shared hold-out)
Classification        → AUC-ROC + Accuracy     (shared hold-out, pooled)
WHY THIS SPLIT
──────────────────────────────────────────────────────────────────────
Course Similarity / User Profile are content-based — there's no
"prediction" to validate against held-out enrollments, so we measure
whether recommendations are actually similar/relevant by construction.
Clustering models don't predict a rating value (they predict a
popularity fraction within a cluster), so RMSE doesn't apply on the
same scale as rating-predicting models. Hit Rate (did ANY held-out
item make top-K) sidesteps the sparsity problem of Precision@K when
a user only has 1-2 held-out items. Silhouette Score is a hold-out-
free structural metric — it tells you if the clusters are well
separated, independent of recommendation quality.
KNN scores a course by the similarity-weighted fraction of the
user's neighbourhood that took it (0-1, implicit feedback), not a
predicted rating — so, like the clustering models, it's graded with
Hit Rate@K rather than RMSE.
NMF / Neural Network / Regression DO predict a continuous
rating-like score, so we evaluate directly against the actual
held-out rating value using RMSE/MAE. This works even with very few
held-out items per user because every (user, item) pair contributes
one data point to a GLOBAL average — no per-user ratio is computed,
avoiding the small-sample problem of Precision@K/Recall@K.
Classification predicts a probability of completion, not a rating,
so RMSE/MAE against a 0/1 label is replaced with AUC-ROC (ranking
quality) and Accuracy (thresholded at 0.5), both POOLED globally
across all held-out + sampled-negative pairs from all users — again
avoiding any per-user sample-size issue.
──────────────────────────────────────────────────────────────────────
SHARED HOLD-OUT SET
──────────────────────────────────────────────────────────────────────
Clustering, Clustering+PCA, KNN, NMF, Neural Network, Regression, and
Classification ALL evaluate on the exact same hold-out split (same
sampled users, same seed, same 20% held-out items per user) so their
scores are directly comparable to each other.
"""

import os
import random
import numpy as np
import pandas as pd
import backend

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
GENRE_PATH   = os.path.join(DATA_DIR, "course_genre.csv")
PROFILE_PATH = os.path.join(DATA_DIR, "user_profile.csv")

GENRE_COLS = [
    "Database", "Python", "CloudComputing", "DataAnalysis", "Containers",
    "MachineLearning", "ComputerVision", "DataScience", "BigData", "Chatbot",
    "R", "BackendDev", "FrontendDev", "Blockchain",
]

DEFAULT_PARAMS = {
    backend.MODELS[0]: {"sim_threshold": 0, "top_courses": 10},
    backend.MODELS[1]: {"profile_threshold": 0, "top_courses": 10},
    backend.MODELS[2]: {"n_clusters": 20, "top_courses": 10},
    backend.MODELS[3]: {"n_clusters": 20, "n_components": 2, "top_courses": 10},
    backend.MODELS[4]: {"n_neighbors": 10, "top_courses": 10},
    backend.MODELS[5]: {"n_components": 15, "top_courses": 10},
    backend.MODELS[6]: {"top_courses": 10},
    backend.MODELS[7]: {"top_courses": 10},
    backend.MODELS[8]: {"top_courses": 10},
}


# ══════════════════════════════════════════════════════════════════════════════
# Shared hold-out split (used by models 2–8)
# ══════════════════════════════════════════════════════════════════════════════

def _build_shared_holdout(n_users: int, test_ratio: float, seed: int):
    """
    Sample n_users with >= 3 enrollments, hold out test_ratio of their
    items. Returns (sampled_users, full_ratings, train_df, test_dict).
    """
    random.seed(seed)
    full_ratings = backend.load_ratings()
    counts   = full_ratings.groupby("user")["item"].count()
    eligible = counts[counts >= 3].index.tolist()
    sampled  = random.sample(eligible, min(n_users, len(eligible)))

    train_rows, test_dict = [], {}
    for uid in sampled:
        items  = full_ratings[full_ratings["user"] == uid]["item"].tolist()
        n_test = max(1, int(len(items) * test_ratio))
        test_items  = random.sample(items, n_test)
        train_items = [i for i in items if i not in test_items]
        test_dict[uid] = test_items
        train_rows.append(
            full_ratings[(full_ratings["user"] == uid) & (full_ratings["item"].isin(train_items))]
        )
    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
    return sampled, full_ratings, train_df, test_dict


def _write_holdout_ratings(full_ratings, sampled_users, train_df):
    """Make load_ratings return the train-only split, IN MEMORY only.

    Sampled users keep just their training items; everyone else keeps all
    their ratings. Nothing is written to ratings.csv on disk, so an
    interrupted eval or a concurrent run can never corrupt the dataset file.
    """
    other = full_ratings[~full_ratings["user"].isin(sampled_users)]
    backend._ratings_override = pd.concat([other, train_df], ignore_index=True)


def _restore_ratings(full_ratings):
    """Clear the in-memory override so load_ratings reads the real CSV again."""
    backend._ratings_override = None


# ── Cache resets so each model retrains cleanly on the hold-out data ───────────

def _reset_knn():
    import models.knn as m
    m._pivot_matrix = None

def _reset_nmf():
    import models.nmf as m
    m._nmf_model = m._reconstructed = m._pivot_columns = m._H_matrix = None
    m._nmf_n_components = None

def _reset_nn():
    import models.neural_network as m
    m._user_embeddings = None
    m._course_embeddings = None

def _reset_regression():
    import models.regression_embedding as m
    m._reg_model = None

def _reset_classification():
    import models.classification_embedding as m
    m._clf_model = None

def _reset_clustering():
    import models.clustering as m
    m._km_model = None; m._km_params = {}
    m._pca_model = None; m._km_pca_model = None; m._pca_params = {}


# ══════════════════════════════════════════════════════════════════════════════
# Model 0 – Course Similarity: Avg Similarity Score (no hold-out)
# ══════════════════════════════════════════════════════════════════════════════

def _eval_course_similarity(n_users: int, seed: int) -> dict:
    random.seed(seed)
    ratings_df = backend.load_ratings()
    eligible   = ratings_df["user"].unique().tolist()
    sampled    = random.sample(eligible, min(n_users, len(eligible)))

    idx_id_dict, id_idx_dict = backend.get_doc_dicts()
    sim_matrix = backend.load_course_sims().to_numpy()
    params     = DEFAULT_PARAMS[backend.MODELS[0]]

    vals = []
    for uid in sampled:
        enrolled = ratings_df[ratings_df["user"] == uid]["item"].tolist()
        res = backend.predict(backend.MODELS[0], [uid], params)
        if res.empty:
            continue
        for rec in res["COURSE_ID"]:
            if rec not in id_idx_dict:
                continue
            max_sim = max(
                (sim_matrix[id_idx_dict[rec]][id_idx_dict[e]]
                 for e in enrolled if e in id_idx_dict),
                default=0.0
            )
            vals.append(max_sim)

    return {"avg_similarity_score": round(float(np.mean(vals)), 4) if vals else 0.0,
            "n_users_evaluated": len(sampled)}


# ══════════════════════════════════════════════════════════════════════════════
# Model 1 – User Profile: Avg Genre Match Score (no hold-out)
# ══════════════════════════════════════════════════════════════════════════════

def _eval_user_profile(n_users: int, seed: int) -> dict:
    random.seed(seed)
    ratings_df = backend.load_ratings()
    eligible   = ratings_df["user"].unique().tolist()
    sampled    = random.sample(eligible, min(n_users, len(eligible)))

    genre_df   = pd.read_csv(GENRE_PATH).set_index("COURSE_ID")[GENRE_COLS]
    profile_df = pd.read_csv(PROFILE_PATH).set_index("user")[GENRE_COLS]
    params     = DEFAULT_PARAMS[backend.MODELS[1]]

    vals = []
    for uid in sampled:
        enrolled = ratings_df[ratings_df["user"] == uid]["item"].tolist()
        if uid in profile_df.index:
            u_vec = profile_df.loc[uid].values.astype(float)
        else:
            in_m  = [c for c in enrolled if c in genre_df.index]
            u_vec = genre_df.loc[in_m].sum(axis=0).values.astype(float) if in_m else np.zeros(len(GENRE_COLS))
        u_norm = u_vec.sum()
        if u_norm == 0:
            continue
        res = backend.predict(backend.MODELS[1], [uid], params)
        if res.empty:
            continue
        for rec in res["COURSE_ID"]:
            if rec in genre_df.index:
                c_vec = genre_df.loc[rec].values.astype(float)
                vals.append(float(np.dot(u_vec, c_vec)) / u_norm)

    return {"avg_genre_match_score": round(float(np.mean(vals)), 4) if vals else 0.0,
            "n_users_evaluated": len(sampled)}


# ══════════════════════════════════════════════════════════════════════════════
# Models 2/3 – Clustering / Clustering+PCA: Hit Rate@K + Silhouette Score
# ══════════════════════════════════════════════════════════════════════════════

def _eval_clustering(model_name: str, sampled, full_ratings, train_df, test_dict,
                     n_clusters: int, k: int, use_pca: bool, n_components: int = 2) -> dict:
    import models.clustering as cl
    from sklearn.metrics import silhouette_score

    _write_holdout_ratings(full_ratings, sampled, train_df)
    _reset_clustering()

    try:
        if use_pca:
            cl.train_pca({"n_clusters": n_clusters, "n_components": n_components})
            X_for_sil = cl._pca_model.transform(cl._pca_params["user_profile_df"].values)
            labels    = cl._pca_params["labels"]
        else:
            cl.train({"n_clusters": n_clusters})
            X_for_sil = cl._km_params["user_profile_df"].values
            labels    = cl._km_params["labels"]

        # Silhouette score needs >= 2 clusters and < n_samples
        try:
            sil = float(silhouette_score(X_for_sil, labels)) if len(set(labels)) > 1 else 0.0
        except Exception:
            sil = 0.0

        hits = []
        params = {**DEFAULT_PARAMS[model_name], "n_clusters": n_clusters, "top_courses": k}
        if use_pca:
            params["n_components"] = n_components

        for uid, test_items in test_dict.items():
            relevant = set(test_items)
            res = backend.predict(model_name, [uid], params)
            if res.empty:
                hits.append(0)
                continue
            rec_ids = res["COURSE_ID"].tolist()[:k]
            hits.append(1 if any(r in relevant for r in rec_ids) else 0)

        return {
            "hit_rate": round(float(np.mean(hits)), 4) if hits else 0.0,
            "silhouette_score": round(sil, 4),
            "n_users_evaluated": len(test_dict),
        }
    finally:
        _restore_ratings(full_ratings)
        _reset_clustering()


# ══════════════════════════════════════════════════════════════════════════════
# Model 4 – KNN: Hit Rate@K
# ══════════════════════════════════════════════════════════════════════════════

def _eval_knn_hit_rate(sampled, full_ratings, train_df, test_dict, k: int) -> dict:
    """
    KNN scores are similarity-weighted neighbourhood fractions (0-1), not
    predicted ratings, so RMSE/MAE against the 2-3 rating scale would be
    meaningless. Grade it the same way as the clustering models instead:
    Hit Rate@K — did ANY held-out course appear in the top-K recommendations.
    """
    _write_holdout_ratings(full_ratings, sampled, train_df)
    _reset_knn()

    try:
        params = {**DEFAULT_PARAMS[backend.MODELS[4]], "top_courses": k}
        hits = []
        for uid, test_items in test_dict.items():
            relevant = set(test_items)
            res = backend.predict(backend.MODELS[4], [uid], params)
            if res.empty:
                hits.append(0)
                continue
            rec_ids = res["COURSE_ID"].tolist()[:k]
            hits.append(1 if any(r in relevant for r in rec_ids) else 0)

        return {"hit_rate": round(float(np.mean(hits)), 4) if hits else 0.0,
                "n_users_evaluated": len(test_dict)}
    finally:
        _restore_ratings(full_ratings)
        _reset_knn()


# ══════════════════════════════════════════════════════════════════════════════
# Models 5/6/7 – NMF / Neural Network / Regression: RMSE + MAE
# ══════════════════════════════════════════════════════════════════════════════

def _eval_rating_predictor(model_name: str, sampled, full_ratings, train_df,
                           test_dict, params: dict, reset_fn) -> dict:
    """
    Computes global RMSE/MAE: for every (user, held-out item) pair where the
    model returns a score for that item, compare predicted score vs actual
    rating (2.0 or 3.0). Pooled across ALL users — avoids per-user sparsity.
    We request scores for the FULL course catalogue (not just top-K) so a
    held-out item that the model ranks poorly still gets included — only
    evaluating top-K here would silently drop exactly the cases where the
    model is wrong, biasing RMSE/MAE optimistically.
    """
    _write_holdout_ratings(full_ratings, sampled, train_df)
    reset_fn()

    try:
        n_courses = len(backend.load_courses())
        errors = []
        n_evaluated_users = 0
        for uid, test_items in test_dict.items():
            full_params = {**params, "top_courses": n_courses}
            res = backend.predict(model_name, [uid], full_params)
            if res.empty:
                continue
            n_evaluated_users += 1
            score_map = dict(zip(res["COURSE_ID"], res["SCORE"]))
            for item in test_items:
                if item in score_map:
                    actual_row = full_ratings[
                        (full_ratings["user"] == uid) & (full_ratings["item"] == item)
                    ]["rating"].values
                    if len(actual_row) > 0:
                        errors.append(score_map[item] - float(actual_row[0]))

        if not errors:
            return {"rmse": None, "mae": None, "n_users_evaluated": n_evaluated_users,
                    "n_pairs_scored": 0}

        errors = np.array(errors)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae  = float(np.mean(np.abs(errors)))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4),
                "n_users_evaluated": n_evaluated_users, "n_pairs_scored": len(errors)}
    finally:
        _restore_ratings(full_ratings)
        reset_fn()


# ══════════════════════════════════════════════════════════════════════════════
# Model 8 – Classification: AUC-ROC + Accuracy (pooled globally)
# ══════════════════════════════════════════════════════════════════════════════

def _eval_classification(sampled, full_ratings, train_df, test_dict, params: dict) -> dict:
    """
    AUC-ROC / Accuracy require the model to score EVERY candidate course,
    not just its top-K — otherwise held-out items that don't make the
    model's top-K are silently dropped from y_true/y_score, which biases
    AUC downward (we'd only ever see the model's most confident, usually
    correct, guesses). We request scores for the full course catalogue
    by setting top_courses to the catalogue size.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    _write_holdout_ratings(full_ratings, sampled, train_df)
    _reset_classification()

    try:
        n_courses = len(backend.load_courses())
        full_params = {**params, "top_courses": n_courses}

        y_true, y_score = [], []
        n_evaluated_users = 0

        for uid, test_items in test_dict.items():
            relevant = set(test_items)
            enrolled_during_eval = set(
                full_ratings[full_ratings["user"] == uid]["item"].tolist()
            ) - relevant   # training enrollments for this user

            res = backend.predict(backend.MODELS[8], [uid], full_params)
            if res.empty:
                continue
            n_evaluated_users += 1

            score_map = dict(zip(res["COURSE_ID"], res["SCORE"]))
            for cid, prob in score_map.items():
                if cid in enrolled_during_eval:
                    continue   # skip items already used as training signal
                y_true.append(1 if cid in relevant else 0)
                y_score.append(prob)

        if len(set(y_true)) < 2:
            return {"auc_roc": None, "accuracy": None,
                    "n_users_evaluated": n_evaluated_users, "n_pairs_scored": len(y_true)}

        auc = float(roc_auc_score(y_true, y_score))
        y_pred = [1 if s >= 0.5 else 0 for s in y_score]
        acc = float(accuracy_score(y_true, y_pred))

        return {"auc_roc": round(auc, 4), "accuracy": round(acc, 4),
                "n_users_evaluated": n_evaluated_users, "n_pairs_scored": len(y_true)}
    finally:
        _restore_ratings(full_ratings)
        _reset_classification()


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all_models(
    n_users:    int   = 100,
    k:          int   = 10,
    test_ratio: float = 0.2,
    seed:       int   = 42,
) -> pd.DataFrame:
    """
    Evaluate all 9 models with ONE metric (or pair) each, completely
    independent of any currently-selected model/params in the UI.
    Models 2–8 share the exact same hold-out split (same users, same
    seed, same held-out items) so their scores are directly comparable.
    Returns a DataFrame, one row per model, NaN for metrics that don't
    apply to that model.
    """
    rows = {}

    # ── No hold-out models ──────────────────────────────────────────────────
    rows[backend.MODELS[0]] = _eval_course_similarity(n_users, seed)
    rows[backend.MODELS[1]] = _eval_user_profile(n_users, seed)

    # ── Shared hold-out split for models 2–8 ───────────────────────────────
    sampled, full_ratings, train_df, test_dict = _build_shared_holdout(
        n_users, test_ratio, seed
    )

    rows[backend.MODELS[2]] = _eval_clustering(
        backend.MODELS[2], sampled, full_ratings, train_df, test_dict,
        n_clusters=20, k=k, use_pca=False
    )
    rows[backend.MODELS[3]] = _eval_clustering(
        backend.MODELS[3], sampled, full_ratings, train_df, test_dict,
        n_clusters=20, k=k, use_pca=True, n_components=2
    )
    rows[backend.MODELS[4]] = _eval_knn_hit_rate(
        sampled, full_ratings, train_df, test_dict, k=k
    )
    rows[backend.MODELS[5]] = _eval_rating_predictor(
        backend.MODELS[5], sampled, full_ratings, train_df, test_dict,
        DEFAULT_PARAMS[backend.MODELS[5]], _reset_nmf
    )
    rows[backend.MODELS[6]] = _eval_rating_predictor(
        backend.MODELS[6], sampled, full_ratings, train_df, test_dict,
        DEFAULT_PARAMS[backend.MODELS[6]], _reset_nn
    )
    rows[backend.MODELS[7]] = _eval_rating_predictor(
        backend.MODELS[7], sampled, full_ratings, train_df, test_dict,
        DEFAULT_PARAMS[backend.MODELS[7]], _reset_regression
    )
    rows[backend.MODELS[8]] = _eval_classification(
        sampled, full_ratings, train_df, test_dict, DEFAULT_PARAMS[backend.MODELS[8]]
    )

    df = pd.DataFrame(rows).T
    df.index.name = "Model"
    return df

"""
Model 2 – Clustering-Based Recommender
Model 3 – Clustering with PCA-Based Recommender
------------------------------------------------
IBM ML Capstone approach:

Data used:
  - user_profile.csv : user × genre matrix (14 genre columns, count values)
  - course_genre.csv : course × genre matrix (binary 0/1)
  - ratings.csv      : user enrollments

How it works (both models):
  1. TRAIN: Run KMeans on the user profile vectors.
             For Model 3, first reduce dimensions with PCA, then cluster.
  2. PREDICT:
     a. Build the new user's profile vector from their selected courses
        (sum of genre vectors, same as User Profile model).
     b. Assign the new user to the nearest cluster.
     c. Find all courses enrolled by users in that cluster.
     d. Rank those courses by enrollment frequency (popularity within cluster).
     e. Remove already-enrolled courses, return top-N.

Corrections vs the original lab notebook:
  - The lab fits KMeans on the full user_profile then predicts a new user's
    cluster using the same fitted model. We do the same but keep the model
    in module-level state so it persists across predict calls without
    re-training every time.
  - Profiles are standardised (StandardScaler) before KMeans/PCA, and the
    same scaler is applied to each user vector at predict time. Without
    this, a new user's small-count vector lands in whatever cluster sits
    nearest the origin regardless of their genre interests.
  - PCA n_components is capped at min(n_samples, n_features)-1 to avoid
    sklearn errors on small datasets.
  - New users not in user_profile.csv get their vector built on the fly
    from course_genre.csv (same approach as User Profile model).
"""

import pandas as pd
import numpy as np
import os
import backend as backend_core

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

GENRE_COLS = [
    "Database", "Python", "CloudComputing", "DataAnalysis", "Containers",
    "MachineLearning", "ComputerVision", "DataScience", "BigData", "Chatbot",
    "R", "BackendDev", "FrontendDev", "Blockchain",
]

# ── Module-level model cache (persists for the process lifetime) ───────────────
_km_model       = None   # KMeans model (Model 2)
_km_params      = {}

_pca_model      = None   # PCA model (Model 3)
_km_pca_model   = None   # KMeans on PCA-reduced data (Model 3)
_pca_params     = {}


# ── Data loaders ───────────────────────────────────────────────────────────────

def _load_user_profiles() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "user_profile.csv"))
    return df.set_index("user")[GENRE_COLS]


def _load_course_genres() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "course_genre.csv"))
    return df.set_index("COURSE_ID")[GENRE_COLS]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _build_new_user_vector(enrolled_ids: list, course_genre_df: pd.DataFrame) -> np.ndarray:
    """Sum genre vectors of enrolled courses to build a profile for a new user."""
    in_matrix = [c for c in enrolled_ids if c in course_genre_df.index]
    if not in_matrix:
        return np.zeros(len(GENRE_COLS))
    return course_genre_df.loc[in_matrix].sum(axis=0).values.astype(float)


def _cluster_recommendations(
    user_vector: np.ndarray,
    cluster_labels: np.ndarray,
    user_profile_df: pd.DataFrame,
    enrolled_ids: list,
    ratings_df: pd.DataFrame,
    top_courses: int,
    transform_fn=None,          # scaling (and PCA for Model 3) applied before predict
    km=None,
) -> dict:
    """
    Core logic shared between Model 2 and Model 3:
    1. Assign user_vector to a cluster.
    2. Find all users in that cluster.
    3. Rank unenrolled courses by how many cluster-members enrolled in them.
    """
    vec = user_vector.reshape(1, -1)
    if transform_fn is not None:
        vec = transform_fn(vec)
    user_cluster = int(km.predict(vec)[0])

    # Users in the same cluster
    cluster_user_ids = user_profile_df.index[cluster_labels == user_cluster].tolist()

    # Course enrollment counts within cluster
    cluster_ratings = ratings_df[ratings_df["user"].isin(cluster_user_ids)]
    course_counts = cluster_ratings["item"].value_counts()

    # Remove already-enrolled courses
    enrolled_set = set(enrolled_ids)
    course_counts = course_counts[~course_counts.index.isin(enrolled_set)]

    if course_counts.empty:
        return {}

    top = course_counts.head(top_courses)
    max_count = top.iloc[0]
    return {cid: round(float(cnt / max_count), 4) for cid, cnt in top.items()}


# ── Model 2 public API ─────────────────────────────────────────────────────────

def train(params: dict = None):
    """Standardise user profile vectors, then fit KMeans."""
    global _km_model, _km_params
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    params = params or {}
    n_clusters = params.get("n_clusters", 20)

    user_profile_df = _load_user_profiles()
    scaler = StandardScaler()
    X = scaler.fit_transform(user_profile_df.values)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    km.fit(X)

    _km_model = km
    _km_params = {"n_clusters": n_clusters, "labels": km.labels_,
                  "user_profile_df": user_profile_df, "scaler": scaler}


def predict(user_ids: list, params: dict = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    user_ids : list of int
    params : dict, optional
        n_clusters  : int   Number of KMeans clusters. Default 20.
        top_courses : int   Max recommendations per user. Default 10.
    """
    global _km_model, _km_params
    from sklearn.cluster import KMeans

    params = params or {}
    n_clusters  = params.get("n_clusters", 20)
    top_courses = params.get("top_courses", 10)

    # Auto-train if not trained yet or cluster count changed
    if _km_model is None or _km_params.get("n_clusters") != n_clusters:
        train({"n_clusters": n_clusters})

    user_profile_df = _km_params["user_profile_df"]
    cluster_labels  = _km_params["labels"]
    course_genre_df = _load_course_genres()
    ratings_df      = backend_core.load_ratings()

    users, courses, scores = [], [], []

    for user_id in user_ids:
        enrolled_ids = ratings_df[ratings_df["user"] == user_id]["item"].tolist()

        # Get or build user vector
        if user_id in user_profile_df.index:
            user_vec = user_profile_df.loc[user_id].values.astype(float)
        else:
            user_vec = _build_new_user_vector(enrolled_ids, course_genre_df)

        if user_vec.sum() == 0:
            continue

        recs = _cluster_recommendations(
            user_vector=user_vec,
            cluster_labels=cluster_labels,
            user_profile_df=user_profile_df,
            enrolled_ids=enrolled_ids,
            ratings_df=ratings_df,
            top_courses=top_courses,
            transform_fn=_km_params["scaler"].transform,
            km=_km_model,
        )

        for course_id, score in recs.items():
            users.append(user_id)
            courses.append(course_id)
            scores.append(score)

    return pd.DataFrame({"USER": users, "COURSE_ID": courses, "SCORE": scores})


# ── Model 3 public API ─────────────────────────────────────────────────────────

def train_pca(params: dict = None):
    """Standardise user profiles, reduce with PCA, then fit KMeans on reduced space."""
    global _pca_model, _km_pca_model, _pca_params
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    params = params or {}
    n_clusters   = params.get("n_clusters", 20)
    n_components = params.get("n_components", 2)   # lab default is 2

    user_profile_df = _load_user_profiles()
    scaler = StandardScaler()
    X = scaler.fit_transform(user_profile_df.values)

    # Cap n_components safely
    max_components = min(X.shape[0], X.shape[1]) - 1
    n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    km.fit(X_reduced)

    _pca_model    = pca
    _km_pca_model = km
    _pca_params   = {
        "n_clusters": n_clusters, "n_components": n_components,
        "labels": km.labels_, "user_profile_df": user_profile_df,
        "scaler": scaler,
    }


def predict_pca(user_ids: list, params: dict = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    user_ids : list of int
    params : dict, optional
        n_clusters   : int   Number of KMeans clusters. Default 20.
        n_components : int   PCA components. Default 2.
        top_courses  : int   Max recommendations per user. Default 10.
    """
    global _pca_model, _km_pca_model, _pca_params

    params = params or {}
    n_clusters   = params.get("n_clusters", 20)
    n_components = params.get("n_components", 2)
    top_courses  = params.get("top_courses", 10)

    # Auto-train if needed or params changed
    if (_pca_model is None
            or _pca_params.get("n_clusters") != n_clusters
            or _pca_params.get("n_components") != n_components):
        train_pca({"n_clusters": n_clusters, "n_components": n_components})

    user_profile_df = _pca_params["user_profile_df"]
    cluster_labels  = _pca_params["labels"]
    course_genre_df = _load_course_genres()
    ratings_df      = backend_core.load_ratings()

    users, courses, scores = [], [], []

    for user_id in user_ids:
        enrolled_ids = ratings_df[ratings_df["user"] == user_id]["item"].tolist()

        if user_id in user_profile_df.index:
            user_vec = user_profile_df.loc[user_id].values.astype(float)
        else:
            user_vec = _build_new_user_vector(enrolled_ids, course_genre_df)

        if user_vec.sum() == 0:
            continue

        scaler = _pca_params["scaler"]
        recs = _cluster_recommendations(
            user_vector=user_vec,
            cluster_labels=cluster_labels,
            user_profile_df=user_profile_df,
            enrolled_ids=enrolled_ids,
            ratings_df=ratings_df,
            top_courses=top_courses,
            transform_fn=lambda v: _pca_model.transform(scaler.transform(v)),
            km=_km_pca_model,
        )

        for course_id, score in recs.items():
            users.append(user_id)
            courses.append(course_id)
            scores.append(score)

    return pd.DataFrame({"USER": users, "COURSE_ID": courses, "SCORE": scores})

import pandas as pd
import os

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = (
    "Course Similarity",                       # 0
    "User Profile",                            # 1
    "Clustering",                              # 2
    "Clustering with PCA",                     # 3
    "KNN",                                     # 4
    "NMF",                                     # 5
    "Neural Network",                          # 6
    "Regression with Embedding Features",      # 7
    "Classification with Embedding Features",  # 8
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Ratings added by live users this session. Kept in memory only — ratings.csv
# on disk is never modified, so repeated clicks and concurrent visitors can't
# corrupt or pollute the dataset file.
_session_ratings = None

# Temporary in-memory replacement for the ratings, used by simple_evaluation
# to feed each model a train-only hold-out split WITHOUT touching the CSV on
# disk. When set, load_ratings returns this verbatim (session rows ignored).
_ratings_override = None


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_ratings() -> pd.DataFrame:
    if _ratings_override is not None:
        return _ratings_override.copy()
    df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    if _session_ratings is not None:
        df = pd.concat([df, _session_ratings], ignore_index=True)
    return df


def load_course_sims() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "sim.csv"))


def load_courses() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "course_processed.csv"))
    df["TITLE"] = df["TITLE"].str.title()
    return df


def load_bow() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "courses_bows.csv"))


# ── Shared helpers ────────────────────────────────────────────────────────────

def get_doc_dicts():
    """Return (idx->id, id->idx) mappings for course similarity lookups."""
    bow_df  = load_bow()
    grouped = bow_df.groupby(["doc_index", "doc_id"]).max().reset_index(drop=False)
    idx_id_dict = grouped[["doc_id"]].to_dict()["doc_id"]
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    return idx_id_dict, id_idx_dict


def add_new_ratings(new_courses) -> int:
    """Register a new user with rating 3.0 for each selected course.

    The rows live in memory only (picked up by load_ratings); the CSV on
    disk is left untouched.
    """
    global _session_ratings
    if len(new_courses) == 0:
        return None
    ratings_df = load_ratings()
    new_id     = int(ratings_df["user"].max()) + 1
    new_rows   = pd.DataFrame({
        "user":   [new_id] * len(new_courses),
        "item":   list(new_courses),
        "rating": [3.0]   * len(new_courses),
    })
    if _session_ratings is None:
        _session_ratings = new_rows
    else:
        _session_ratings = pd.concat([_session_ratings, new_rows], ignore_index=True)
    return new_id


# ── Model dispatch ────────────────────────────────────────────────────────────

def train(model_name: str, params: dict = None):
    params = params or {}
    if model_name == MODELS[0]:
        pass  # Course Similarity: no training
    elif model_name == MODELS[1]:
        pass  # User Profile: no training
    elif model_name == MODELS[2]:
        from models.clustering import train as _train
        _train(params)
    elif model_name == MODELS[3]:
        from models.clustering import train_pca
        train_pca(params)
    elif model_name == MODELS[4]:
        from models.knn import train as _train
        _train(params)
    elif model_name == MODELS[5]:
        from models.nmf import train as _train
        _train(params)
    elif model_name == MODELS[6]:
        from models.neural_network import train as _train
        _train(params)
    elif model_name == MODELS[7]:
        from models.regression_embedding import train as _train
        _train(params)
    elif model_name == MODELS[8]:
        from models.classification_embedding import train as _train
        _train(params)


def predict(model_name: str, user_ids: list, params: dict = None) -> pd.DataFrame:
    params = params or {}

    if model_name == MODELS[0]:
        from models.course_similarity import predict as _predict
        return _predict(user_ids, params)
    elif model_name == MODELS[1]:
        from models.user_profile import predict as _predict
        return _predict(user_ids, params)
    elif model_name == MODELS[2]:
        from models.clustering import predict as _predict
        return _predict(user_ids, params)
    elif model_name == MODELS[3]:
        from models.clustering import predict_pca as _predict
        return _predict(user_ids, params)
    elif model_name == MODELS[4]:
        from models.knn import predict as _predict
        return _predict(user_ids, params)
    elif model_name == MODELS[5]:
        from models.nmf import predict as _predict
        return _predict(user_ids, params)
    elif model_name == MODELS[6]:
        from models.neural_network import predict as _predict
        return _predict(user_ids, params)
    elif model_name == MODELS[7]:
        from models.regression_embedding import predict as _predict
        return _predict(user_ids, params)
    elif model_name == MODELS[8]:
        from models.classification_embedding import predict as _predict
        return _predict(user_ids, params)

    return pd.DataFrame(columns=["USER", "COURSE_ID", "SCORE"])

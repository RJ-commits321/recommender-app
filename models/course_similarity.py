"""
Model 0 – Course Similarity
----------------------------
Recommends courses whose content is most similar to the courses a user has
already enrolled in, using a pre-computed cosine-similarity matrix (sim.csv).

No training step is required — the similarity matrix is loaded directly.
"""

import pandas as pd
import backend as backend_core


# ── Core recommendation logic ─────────────────────────────────────────────────

def _course_similarity_recommendations(
    idx_id_dict: dict,
    id_idx_dict: dict,
    enrolled_course_ids: list,
    sim_matrix,
) -> dict:
    """
    For every enrolled course, score every unenrolled course by its similarity.
    Keeps the *maximum* similarity score across all enrolled courses.

    Returns
    -------
    dict  {course_id: similarity_score}, sorted descending.
    """
    all_courses = set(idx_id_dict.values())
    unselected = all_courses.difference(set(enrolled_course_ids))

    scores = {}
    for enrolled in enrolled_course_ids:
        for candidate in unselected:
            if enrolled not in id_idx_dict or candidate not in id_idx_dict:
                continue
            sim = sim_matrix[id_idx_dict[enrolled]][id_idx_dict[candidate]]
            if candidate not in scores or sim > scores[candidate]:
                scores[candidate] = sim

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# ── Public API ────────────────────────────────────────────────────────────────

def train(params: dict = None):
    """No-op: Course Similarity requires no model training."""
    pass


def predict(user_ids: list, params: dict = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    user_ids : list of int
        Users to generate recommendations for.
    params : dict, optional
        sim_threshold : int   Minimum similarity % (0-100). Default 60.
        top_courses   : int   Maximum recommendations per user. Default 10.

    Returns
    -------
    pd.DataFrame with columns [USER, COURSE_ID, SCORE]
    """
    params = params or {}
    sim_threshold = params.get("sim_threshold", 60) / 100.0
    top_courses   = params.get("top_courses", 10)

    idx_id_dict, id_idx_dict = backend_core.get_doc_dicts()
    sim_matrix = backend_core.load_course_sims().to_numpy()
    ratings_df = backend_core.load_ratings()

    users, courses, scores = [], [], []

    for user_id in user_ids:
        enrolled_ids = (
            ratings_df[ratings_df["user"] == user_id]["item"].tolist()
        )
        recs = _course_similarity_recommendations(
            idx_id_dict, id_idx_dict, enrolled_ids, sim_matrix
        )
        count = 0
        for course_id, score in recs.items():
            if score < sim_threshold:
                break
            if count >= top_courses:
                break
            users.append(user_id)
            courses.append(course_id)
            scores.append(round(score, 4))
            count += 1

    return pd.DataFrame({"USER": users, "COURSE_ID": courses, "SCORE": scores})

"""
Model 1 – User Profile (Content-Based via Course Genres)
----------------------------------------------------------
IBM ML Capstone approach:

Data files used:
  - course_genre.csv  : COURSE_ID, TITLE, + 14 genre columns (binary 0/1)
  - user_profile.csv  : user, + 14 genre columns (summed enrollment counts)

How it works:
  1. For a new user, build their profile vector by summing the genre vectors
     of every course they selected (same 14 genre columns as course_genre.csv).
  2. For every unenrolled course, compute an interest score as the dot product
     of the user profile vector and the course's genre vector.
  3. Normalise scores to [0, 1], filter by threshold, return top-N.

No training step required — scores are computed on the fly.
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


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_course_genres() -> pd.DataFrame:
    """Returns course_genre.csv with COURSE_ID as index, genre columns as values."""
    df = pd.read_csv(os.path.join(DATA_DIR, "course_genre.csv"))
    df = df.set_index("COURSE_ID")[GENRE_COLS]
    return df


def load_user_profiles() -> pd.DataFrame:
    """Returns user_profile.csv with user as index, genre columns as values."""
    df = pd.read_csv(os.path.join(DATA_DIR, "user_profile.csv"))
    df = df.set_index("user")[GENRE_COLS]
    return df


# ── Core logic ─────────────────────────────────────────────────────────────────

def _build_profile_for_new_user(enrolled_course_ids: list, course_genre_df: pd.DataFrame) -> np.ndarray:
    """
    Build a profile vector for a new user by summing the genre vectors
    of their enrolled courses. Courses absent from the genre matrix are skipped.
    """
    enrolled_in_matrix = [c for c in enrolled_course_ids if c in course_genre_df.index]
    if not enrolled_in_matrix:
        return np.zeros(len(GENRE_COLS))
    return course_genre_df.loc[enrolled_in_matrix].sum(axis=0).values.astype(float)


def _score_courses(user_profile_vector: np.ndarray, enrolled_ids: list,
                   course_genre_df: pd.DataFrame) -> dict:
    """
    Score every unenrolled course by dot product with the user profile vector.
    Returns {course_id: raw_score} for scores > 0.
    """
    enrolled_set = set(enrolled_ids)
    scores = {}
    for course_id, genre_row in course_genre_df.iterrows():
        if course_id in enrolled_set:
            continue
        score = float(np.dot(user_profile_vector, genre_row.values))
        if score > 0:
            scores[course_id] = score
    return scores


# ── Public API ─────────────────────────────────────────────────────────────────

def train(params: dict = None):
    """No-op: User Profile requires no model training."""
    pass


def predict(user_ids: list, params: dict = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    user_ids : list of int
    params : dict, optional
        profile_threshold : int   Min interest score as % of max (0-100). Default 0.
        top_courses       : int   Max recommendations per user. Default 10.

    Returns
    -------
    pd.DataFrame  columns: [USER, COURSE_ID, SCORE]
    """
    params = params or {}
    top_courses = params.get("top_courses", 10)
    threshold_pct = params.get("profile_threshold", 0) / 100.0

    course_genre_df = load_course_genres()
    user_profile_df = load_user_profiles()
    ratings_df = backend_core.load_ratings()

    users, courses, scores_out = [], [], []

    for user_id in user_ids:
        enrolled_ids = ratings_df[ratings_df["user"] == user_id]["item"].tolist()

        # Use pre-computed profile if available, otherwise build from enrollments
        if user_id in user_profile_df.index:
            user_vector = user_profile_df.loc[user_id].values.astype(float)
        else:
            user_vector = _build_profile_for_new_user(enrolled_ids, course_genre_df)

        if user_vector.sum() == 0:
            continue

        # Score all unenrolled courses
        raw_scores = _score_courses(user_vector, enrolled_ids, course_genre_df)
        if not raw_scores:
            continue

        # Normalise to [0, 1]
        max_score = max(raw_scores.values())
        normalised = {cid: s / max_score for cid, s in raw_scores.items()}

        # Filter by threshold, sort descending, take top-N
        filtered = {cid: s for cid, s in normalised.items() if s >= threshold_pct}
        top = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_courses]

        for course_id, score in top:
            users.append(user_id)
            courses.append(course_id)
            scores_out.append(round(score, 4))

    return pd.DataFrame({"USER": users, "COURSE_ID": courses, "SCORE": scores_out})

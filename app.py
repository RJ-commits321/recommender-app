import streamlit as st
import pandas as pd
import backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

st.set_page_config(
    page_title="Course Recommender System",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached loaders ─────────────────────────────────────────────────────────────
# Only the course catalogue is cached: it's the one dataset the UI itself
# renders (grid + results merge). The models call backend.load_* directly,
# so caching ratings/sims/bow here would buy nothing — and a cached ratings
# copy would go stale as soon as a session user is added.
@st.cache_data
def load_courses():
    return backend.load_courses()


# ── App init ───────────────────────────────────────────────────────────────────
def init_recommender_app():
    with st.spinner("Loading course catalogue..."):
        course_df = load_courses()

    st.success("Course catalogue loaded.")
    st.markdown("---")
    st.subheader("Select courses that you have audited or completed:")

    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    grid_options = gb.build()

    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(
        response["selected_rows"], columns=["COURSE_ID", "TITLE", "DESCRIPTION"]
    )[["COURSE_ID", "TITLE"]]

    st.subheader("Your selected courses:")
    st.table(results)
    return results


# ── Train / Predict helpers ────────────────────────────────────────────────────
def train(model_name, params):
    with st.spinner("Training..."):
        backend.train(model_name, params)
    st.success("Training complete!")


def predict(model_name, user_ids, params):
    with st.spinner("Generating recommendations..."):
        res = backend.predict(model_name, user_ids, params)
    st.success("Recommendations generated!")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("Personalized Learning Recommender")
selected_courses_df = init_recommender_app()

st.sidebar.subheader("1. Select model")
model_selection = st.sidebar.selectbox("Select model:", backend.MODELS)

st.sidebar.subheader("2. Hyper-parameters")
params = {}

if model_selection == backend.MODELS[0]:
    params["top_courses"]   = st.sidebar.slider("Top courses", 1, 50, 10)
    params["sim_threshold"] = st.sidebar.slider("Similarity threshold %", 0, 100, 50, 10)
elif model_selection == backend.MODELS[1]:
    params["top_courses"]       = st.sidebar.slider("Top courses", 1, 50, 10)
    params["profile_threshold"] = st.sidebar.slider("Profile threshold %", 0, 100, 0, 10)
elif model_selection == backend.MODELS[2]:
    params["n_clusters"]  = st.sidebar.slider("Clusters", 2, 50, 20)
    params["top_courses"] = st.sidebar.slider("Top courses", 1, 50, 10)
elif model_selection == backend.MODELS[3]:
    params["n_clusters"]   = st.sidebar.slider("Clusters", 2, 50, 20)
    params["n_components"] = st.sidebar.slider("PCA components", 2, 14, 2)
    params["top_courses"]  = st.sidebar.slider("Top courses", 1, 50, 10)
elif model_selection == backend.MODELS[4]:
    params["n_neighbors"] = st.sidebar.slider("Neighbours", 2, 50, 10)
    params["top_courses"] = st.sidebar.slider("Top courses", 1, 50, 10)
elif model_selection == backend.MODELS[5]:
    params["n_components"] = st.sidebar.slider("Latent factors", 2, 50, 15)
    params["top_courses"]  = st.sidebar.slider("Top courses", 1, 50, 10)
elif model_selection == backend.MODELS[6]:
    params["epochs"]      = st.sidebar.slider("Epochs", 5, 50, 10)
    params["top_courses"] = st.sidebar.slider("Top courses", 1, 50, 10)
elif model_selection == backend.MODELS[7]:
    params["top_courses"] = st.sidebar.slider("Top courses", 1, 50, 10)
elif model_selection == backend.MODELS[8]:
    params["top_courses"] = st.sidebar.slider("Top courses", 1, 50, 10)

st.sidebar.subheader("3. Train")
# Course Similarity and User Profile score on the fly — there is nothing to
# fit, so showing a Train button (and a "Training complete!" toast) for them
# would be misleading.
if model_selection in (backend.MODELS[0], backend.MODELS[1]):
    st.sidebar.caption("This model needs no training — go straight to Predict.")
else:
    if st.sidebar.button("Train Model"):
        train(model_selection, params)

st.sidebar.subheader("4. Predict")
if st.sidebar.button("Recommend New Courses"):
    if selected_courses_df.shape[0] == 0:
        st.warning("Please select at least one course first.")
        st.session_state.pop("recommendations", None)
    else:
        # Reuse this browser session's user id so repeated clicks refresh the
        # same user instead of piling up a new phantom user each time.
        new_id = backend.add_new_ratings(
            selected_courses_df["COURSE_ID"].values,
            user_id=st.session_state.get("session_user_id"),
        )
        st.session_state["session_user_id"] = new_id
        st.session_state["recommendations"] = predict(model_selection, [new_id], params)
        st.session_state["rec_model"] = model_selection

# Rendered outside the button block so results survive the rerun that any
# widget change triggers (moving a slider would otherwise blank the table).
_res_df = st.session_state.get("recommendations")
if _res_df is not None:
    if _res_df.empty:
        st.warning("No recommendations found. Try adjusting hyper-parameters.")
    else:
        _out = _res_df[["COURSE_ID", "SCORE"]]
        _out = pd.merge(_out, load_courses(), on="COURSE_ID").drop("COURSE_ID", axis=1)
        st.subheader(f"Recommended Courses — {st.session_state.get('rec_model', '')}")
        st.table(_out)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION  (main page expander)
# ──────────────────────────────────────────────────────────────────────────────
# TO DISABLE: comment out this entire block (or delete simple_evaluation.py
# and remove the "import simple_evaluation" call below). Nothing else in
# the app depends on it.
# ══════════════════════════════════════════════════════════════════════════════
# st.markdown("---")
# with st.expander("📊 Evaluate All 9 Models", expanded=False):
#     st.write(
#         "Tests all 9 models at once, separately from whatever model you have "
#         "selected in the sidebar. Each model is graded with the metric(s) "
#         "that actually fit how it makes predictions:"
#     )

#     reference_df = pd.DataFrame([
#         ["Course Similarity",                       "Avg Similarity",        "—"],
#         ["User Profile",                             "Avg Genre Match",       "—"],
#         ["Clustering",                               "Hit Rate@K, Silhouette", "Top-K slider"],
#         ["Clustering with PCA",                      "Hit Rate@K, Silhouette", "Top-K slider"],
#         ["KNN",                                      "Hit Rate@K",            "Top-K slider"],
#         ["NMF",                                      "Hit Rate@K",            "Top-K slider"],
#         ["Neural Network",                           "RMSE, MAE",             "—"],
#         ["Regression with Embedding Features",       "RMSE, MAE",             "—"],
#         ["Classification with Embedding Features",   "AUC-ROC, Accuracy",     "—"],
#     ], columns=["Model", "Metric(s) used", "Uses Top-K slider?"])
#     st.dataframe(reference_df, hide_index=True, use_container_width=True)

#     st.write(" ")
#     eval_n_users = st.slider(
#         "Users to sample",
#         30, 300, 100, 10, key="simple_eval_n",
#         help="Number of random users tested. The SAME sample is used for every "
#              "model below, so their results are directly comparable to each other."
#     )
#     eval_k = st.slider(
#         "Top-K",
#         5, 20, 10, key="simple_eval_k",
#         help="Only affects the Hit Rate metric of Clustering, Clustering+PCA, "
#              "KNN and NMF (checks if any held-out course appears in the top K "
#              "recommendations). All other models below ignore this setting."
#     )

#     if st.button("Run Evaluation", key="run_simple_eval"):
#         import simple_evaluation as sev   # comment out this import to disable

#         with st.spinner(
#             f"Evaluating all 9 models on {eval_n_users} users — this can take "
#             "a couple of minutes since classification/regression score the "
#             "full course catalogue for accurate AUC/RMSE..."
#         ):
#             results_df = sev.evaluate_all_models(n_users=eval_n_users, k=eval_k)

#         st.write(" ")
#         st.subheader("Results")

#         metric_label = {
#             "avg_similarity_score":   "Avg Similarity",
#             "avg_genre_match_score":  "Avg Genre Match",
#             "hit_rate":               "Hit Rate@K",
#             "silhouette_score":       "Silhouette",
#             "rmse":                   "RMSE",
#             "mae":                    "MAE",
#             "auc_roc":                "AUC-ROC",
#             "accuracy":               "Accuracy",
#         }

#         display_cols = [c for c in metric_label if c in results_df.columns]
#         display_df = results_df[display_cols].rename(columns=metric_label)
#         st.dataframe(display_df.style.format(precision=4, na_rep="—"),
#                     use_container_width=True)

#         st.caption(
#             "Blank cells (—) mean that metric doesn't apply to that model. "
#             "Clustering, KNN, NMF, Neural Network, Regression, and Classification "
#             "were all tested on the exact same hold-out users/items, so their "
#             "scores can be compared row-to-row."
#         )

#         st.caption(
#             "Course Similarity / User Profile: no hold-out, measured directly. "
#             "Clustering / Clustering+PCA: Hit Rate@K + Silhouette (structural). "
#             "KNN / NMF: Hit Rate@K (scores are normalized 0-1, not ratings). "
#             "Neural Network / Regression: RMSE + MAE vs actual rating. "
#             "Classification: AUC-ROC + Accuracy (probability of completion)."
#         )

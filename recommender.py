# streamlit_recommender_viz.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Item-Based Recommender with Visuals", layout="wide")
st.title("📊 Item-Based Collaborative Filtering Recommender System")

# =====================================
# 1. Load or generate ratings dataset
# =====================================
def load_or_generate_ratings(
    filename="ratings.csv",
    num_users=50,
    num_items=30,
    min_ratings=5,
    max_ratings=15,
    seed=42
):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    np.random.seed(seed)
    data = []
    for user_id in range(1, num_users + 1):
        n = np.random.randint(min_ratings, max_ratings + 1)
        items = np.random.choice(range(101, 101 + num_items), n, replace=False)
        for item_id in items:
            rating = np.random.randint(1, 6)
            data.append([user_id, item_id, rating])
    df = pd.DataFrame(data, columns=["user_id", "item_id", "rating"])
    df.to_csv(filename, index=False)
    return df

uploaded_file = st.file_uploader("Upload ratings CSV (optional)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_or_generate_ratings()

st.write("Sample of dataset:", df.head())

# =====================================
# 2. Train/test split
# =====================================
def train_test_split_by_user(df, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    train, test = [], []
    for user_id, group in df.groupby("user_id"):
        n_test = max(1, int(len(group) * test_ratio))
        test_idx = np.random.choice(group.index, n_test, replace=False)
        test.append(group.loc[test_idx])
        train.append(group.drop(test_idx))
    return pd.concat(train).reset_index(drop=True), pd.concat(test).reset_index(drop=True)

train_df, test_df = train_test_split_by_user(df)

# =====================================
# 3. User–item matrix and similarity
# =====================================
user_item_matrix = train_df.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0)
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# =====================================
# 4. Recommendation function
# =====================================
def recommend_items_item_based(user_id, user_item_matrix, item_similarity_df, k=5, min_similarity=0.2):
    if user_id not in user_item_matrix.index:
        return []
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0]
    scores = {}
    for item, rating in rated_items.items():
        similar_items = item_similarity_df[item]
        similar_items = similar_items[similar_items >= min_similarity]
        for sim_item, sim in similar_items.items():
            if sim_item in rated_items:
                continue
            scores[sim_item] = scores.get(sim_item, 0) + sim * rating
    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked_items[:k]]

# =====================================
# 5. Precision@K evaluation
# =====================================
def precision_at_k(user_item_matrix, test_df, item_similarity_df, k=5, min_similarity=0.2):
    precisions = []
    for user_id in test_df["user_id"].unique():
        true_items = set(test_df[test_df["user_id"] == user_id]["item_id"])
        recommended = recommend_items_item_based(user_id, user_item_matrix, item_similarity_df, k, min_similarity)
        if not recommended:
            continue
        hits = len(set(recommended) & true_items)
        precisions.append(hits / k)
    return sum(precisions) / len(precisions) if precisions else 0

p_at_5 = precision_at_k(user_item_matrix, test_df, item_similarity_df)
st.metric("Precision@5", f"{p_at_5:.4f}")

# =====================================
# 6. Interactive recommendations
# =====================================
st.subheader("Get Recommendations for a User")
user_id_input = st.number_input("Enter User ID:", min_value=int(df.user_id.min()), max_value=int(df.user_id.max()), value=int(df.user_id.min()))
k_input = st.slider("Number of recommendations (K)", min_value=1, max_value=10, value=5)
min_sim_input = st.slider("Minimum similarity threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

if st.button("Recommend"):
    recs = recommend_items_item_based(user_id_input, user_item_matrix, item_similarity_df, k=k_input, min_similarity=min_sim_input)
    if recs:
        st.write(f"Top {k_input} recommended items for user {user_id_input}: {recs}")
    else:
        st.write(f"No recommendations could be generated for user {user_id_input}.")

# =====================================
# 7. Visualizations
# =====================================

st.subheader("Item–Item Similarity Heatmap")
heatmap_fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(item_similarity_df, cmap="coolwarm", ax=ax)
st.pyplot(heatmap_fig)

st.subheader("Top-N Most Rated Items Globally")
top_items = df.groupby("item_id")["rating"].count().sort_values(ascending=False).head(10)
st.bar_chart(top_items)

# app_minimal.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Item-Based Recommender", layout="wide")
st.title("📊 Minimal Item-Based Recommender (Streamlit + Pandas Only)")

# ------------------------
# 1. Load or generate dataset
# ------------------------
def generate_ratings(num_users=20, num_items=15):
    data = []
    for user_id in range(1, num_users + 1):
        items = list(range(101, 101 + num_items))
        for item_id in items:
            rating = (user_id + item_id) % 5 + 1  # simple pattern for demo
            data.append([user_id, item_id, rating])
    return pd.DataFrame(data, columns=["user_id", "item_id", "rating"])

uploaded_file = st.file_uploader("Upload ratings CSV (optional)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = generate_ratings()

st.write("Sample data:", df.head())

# ------------------------
# 2. User-item matrix
# ------------------------
user_item = df.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0)

# ------------------------
# 3. Simple item–item similarity (using pandas)
# ------------------------
# Cosine similarity formula: sim(A,B) = (A*B).sum() / (sqrt(A^2).sum() * sqrt(B^2).sum())
def cosine_similarity_pandas(matrix):
    items = matrix.columns
    similarity = pd.DataFrame(index=items, columns=items, dtype=float)
    for i in items:
        for j in items:
            vec_i = matrix[i]
            vec_j = matrix[j]
            num = (vec_i * vec_j).sum()
            den = (vec_i**2).sum()**0.5 * (vec_j**2).sum()**0.5
            similarity.loc[i,j] = num / den if den != 0 else 0
    return similarity

item_similarity = cosine_similarity_pandas(user_item)

# ------------------------
# 4. Recommend items for a user
# ------------------------
def recommend_items(user_id, k=5, min_sim=0.2):
    if user_id not in user_item.index:
        return []
    user_ratings = user_item.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index.tolist()
    scores = {}
    for item in rated_items:
        sims = item_similarity[item]
        sims = sims[sims >= min_sim]
        for sim_item, sim_val in sims.items():
            if sim_item in rated_items:
                continue
            scores[sim_item] = scores.get(sim_item,0) + sim_val * user_ratings[item]
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item,_ in ranked[:k]]

# ------------------------
# 5. Interactive recommendations
# ------------------------
st.subheader("Get Recommendations for a User")
user_input = st.number_input("User ID:", min_value=int(df.user_id.min()), max_value=int(df.user_id.max()), value=int(df.user_id.min()))
k_input = st.slider("Number of recommendations (K)", min_value=1, max_value=10, value=5)
min_sim_input = st.slider("Minimum similarity threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

if st.button("Recommend"):
    recs = recommend_items(user_input, k_input, min_sim_input)
    if recs:
        st.write(f"Top {k_input} recommendations for user {user_input}: {recs}")
    else:
        st.write(f"No recommendations available for user {user_input}.")

# ------------------------
# 6. Top-N most rated items
# ------------------------
st.subheader("Top-N Most Rated Items Globally")
top_items = df.groupby("item_id")["rating"].count().sort_values(ascending=False).head(10)
st.bar_chart(top_items)

# ------------------------
# 7. Item–item similarity preview
# ------------------------
st.subheader("Item–Item Similarity Preview (first 10 items)")
st.dataframe(item_similarity.iloc[:10, :10])

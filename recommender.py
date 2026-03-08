import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Product Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #FF9900 0%, #FF6600 100%);
        padding: 2rem; border-radius: 14px; color: white;
        text-align: center; margin-bottom: 1.5rem;
    }
    .main-header h1 { font-size: 2.4rem; margin: 0; }
    .main-header p  { font-size: 1rem; margin: 0.4rem 0 0; opacity: 0.92; }
    .section-header {
        color: #FF6600; border-bottom: 2px solid #FF9900;
        padding-bottom: 0.3rem; margin: 1.5rem 0 1rem;
        font-size: 1.2rem; font-weight: 700;
    }
    .product-card {
        background: #fff8f0;
        border: 1px solid #FFD699;
        border-left: 5px solid #FF9900;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .product-card h4 { margin: 0 0 0.3rem; color: #232F3E; font-size: 0.97rem; }
    .badge {
        display: inline-block; background: #FF9900; color: white;
        border-radius: 4px; padding: 2px 8px; font-size: 0.78rem;
        font-weight: 600; margin-right: 4px;
    }
    .badge-green { background: #2ecc71; }
    .badge-blue  { background: #3498db; }
    .stButton > button {
        background: linear-gradient(135deg, #FF9900, #FF6600);
        color: white; border: none; font-weight: 700;
        border-radius: 8px; padding: 0.55rem 1.8rem; width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; }
    .sim-bar-outer {
        background: #eee; border-radius: 6px; height: 10px; width: 100%;
    }
    .sim-bar-inner {
        background: linear-gradient(90deg, #FF9900, #FF6600);
        border-radius: 6px; height: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading & preprocessing ───────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(raw_bytes):
    import io
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Clean prices
    for col in ["discounted_price", "actual_price"]:
        df[col] = df[col].astype(str).str.replace("₹", "").str.replace(",", "").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean discount & rating
    df["discount_percentage"] = df["discount_percentage"].astype(str).str.replace("%", "").str.strip()
    df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count"] = df["rating_count"].astype(str).str.replace(",", "").str.strip()
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")

    # Main category
    df["main_category"] = df["category"].str.split("|").str[0]
    df["sub_category"]  = df["category"].str.split("|").str[1].fillna("")

    # Drop duplicates on product_id
    df = df.drop_duplicates(subset="product_id").reset_index(drop=True)

    # Fill nulls
    df["rating"].fillna(df["rating"].median(), inplace=True)
    df["rating_count"].fillna(0, inplace=True)
    df["about_product"].fillna("", inplace=True)
    df["review_title"].fillna("", inplace=True)
    df["review_content"].fillna("", inplace=True)

    # Combined text for content-based filtering
    df["combined_text"] = (
        df["product_name"].fillna("") + " " +
        df["main_category"].fillna("") + " " +
        df["sub_category"].fillna("") + " " +
        df["about_product"].fillna("") + " " +
        df["review_title"].fillna("")
    )

    # Weighted score for ranking
    v = df["rating_count"]
    m = v.quantile(0.25)
    C = df["rating"].mean()
    df["weighted_score"] = (
        (v / (v + m)) * df["rating"] + (m / (v + m)) * C
    ).round(3)

    return df


@st.cache_resource
def build_tfidf(combined_texts):
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2),
        min_df=1
    )
    matrix = tfidf.fit_transform(combined_texts)
    return tfidf, matrix


def get_content_recommendations(df, tfidf_matrix, product_idx, top_n=10):
    sim_scores = cosine_similarity(tfidf_matrix[product_idx], tfidf_matrix).flatten()
    sim_scores[product_idx] = 0
    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    results = df.iloc[top_indices].copy()
    results["similarity"] = sim_scores[top_indices]
    return results


def get_category_recommendations(df, category, top_n=10, sort_by="weighted_score"):
    filtered = df[df["main_category"] == category].copy()
    if sort_by == "weighted_score":
        filtered = filtered.sort_values("weighted_score", ascending=False)
    elif sort_by == "rating":
        filtered = filtered.sort_values("rating", ascending=False)
    elif sort_by == "discount_percentage":
        filtered = filtered.sort_values("discount_percentage", ascending=False)
    elif sort_by == "discounted_price":
        filtered = filtered.sort_values("discounted_price", ascending=True)
    return filtered.head(top_n)


def get_price_range_recommendations(df, min_price, max_price, category=None, top_n=10):
    filtered = df[
        (df["discounted_price"] >= min_price) &
        (df["discounted_price"] <= max_price)
    ].copy()
    if category and category != "All":
        filtered = filtered[filtered["main_category"] == category]
    return filtered.sort_values("weighted_score", ascending=False).head(top_n)


def render_product_card(row, rank=None, show_similarity=False):
    prefix = f"#{rank} " if rank else ""
    disc_str = f"₹{row['discounted_price']:,.0f}" if pd.notna(row['discounted_price']) else "N/A"
    act_str  = f"₹{row['actual_price']:,.0f}"    if pd.notna(row['actual_price'])    else ""
    disc_pct = f"{row['discount_percentage']:.0f}%" if pd.notna(row['discount_percentage']) else ""
    rating   = f"⭐ {row['rating']:.1f}" if pd.notna(row['rating']) else "N/A"
    rcount   = f"{int(row['rating_count']):,} ratings" if pd.notna(row['rating_count']) and row['rating_count'] > 0 else ""
    name     = str(row['product_name'])[:110] + ("…" if len(str(row['product_name'])) > 110 else "")
    cat      = str(row['main_category'])

    sim_html = ""
    if show_similarity and 'similarity' in row and pd.notna(row['similarity']):
        pct = int(row['similarity'] * 100)
        sim_html = f"""
        <div style='margin-top:6px;'>
          <span style='font-size:0.78rem;color:#888;'>Match: {pct}%</span>
          <div class='sim-bar-outer'><div class='sim-bar-inner' style='width:{pct}%'></div></div>
        </div>"""

    link = str(row.get('product_link', ''))
    link_html = f"<a href='{link}' target='_blank' style='font-size:0.78rem;color:#FF6600;'>View on Amazon ↗</a>" if link and link.startswith("http") else ""

    st.markdown(f"""
    <div class='product-card'>
      <h4>{prefix}{name}</h4>
      <span class='badge'>{cat}</span>
      <span class='badge badge-green'>{rating}</span>
      {"<span class='badge badge-blue'>" + disc_pct + " off</span>" if disc_pct else ""}
      <br><br>
      <span style='font-size:1rem;font-weight:700;color:#B12704;'>{disc_str}</span>
      {"&nbsp;<span style='font-size:0.82rem;color:#888;text-decoration:line-through;'>" + act_str + "</span>" if act_str else ""}
      &nbsp;&nbsp;<span style='font-size:0.82rem;color:#555;'>{rcount}</span>
      {sim_html}
      <br>{link_html}
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=130)
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload amazon.csv", type=["csv"])
    st.markdown("---")
    st.subheader("⚙️ Settings")
    top_n = st.slider("Results to show", 5, 20, 10)
    st.markdown("---")
    st.caption("🛒 Amazon Product Recommender · v1.0")


# ── Gate on upload ─────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class='main-header'>
        <h1>🛒 Amazon Product Recommender</h1>
        <p>Content-Based · Category-Based · Price-Filtered · EDA Dashboard</p>
    </div>""", unsafe_allow_html=True)
    st.info("👈 Upload your **amazon.csv** file in the sidebar to get started.")
    st.stop()

df = load_and_preprocess(uploaded.getvalue())
_, tfidf_matrix = build_tfidf(df["combined_text"].tolist())

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🛒 Amazon Product Recommender</h1>
    <p>Discover similar products, top-rated picks, best deals, and more</p>
</div>""", unsafe_allow_html=True)

# ── KPIs ───────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📦 Products",       f"{len(df):,}")
k2.metric("🏷️ Categories",     df["main_category"].nunique())
k3.metric("⭐ Avg Rating",      f"{df['rating'].mean():.2f}")
k4.metric("💰 Avg Discount",    f"{df['discount_percentage'].mean():.1f}%")
k5.metric("📝 Avg Reviews",     f"{df['rating_count'].median():,.0f}")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Content-Based",
    "🏆 Category Top Picks",
    "💸 Price Filter",
    "📊 EDA Dashboard",
    "🗂️ Data Explorer"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CONTENT-BASED RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🔍 Find Similar Products</div>', unsafe_allow_html=True)
    st.write("Search for a product and get recommendations based on name, category, description & reviews.")

    search_query = st.text_input("🔎 Search product name", placeholder="e.g. boAt headphones, USB cable, smart bulb…")

    if search_query:
        mask = df["product_name"].str.contains(search_query, case=False, na=False)
        results = df[mask]
        if results.empty:
            st.warning("No products found. Try a different keyword.")
        else:
            st.success(f"Found **{len(results)}** matching product(s). Select one to get recommendations:")
            selected_name = st.selectbox("Select a product", results["product_name"].tolist())
            selected_row  = df[df["product_name"] == selected_name].iloc[0]
            product_idx   = df[df["product_name"] == selected_name].index[0]

            st.markdown("#### 📌 Selected Product")
            render_product_card(selected_row)

            if st.button("🚀 Get Recommendations"):
                recs = get_content_recommendations(df, tfidf_matrix, product_idx, top_n)
                st.markdown(f"#### 🎯 Top {len(recs)} Similar Products")
                for rank, (_, row) in enumerate(recs.iterrows(), 1):
                    render_product_card(row, rank=rank, show_similarity=True)
    else:
        st.info("Start typing a product name above to search.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CATEGORY-BASED TOP PICKS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🏆 Top Products by Category</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        categories = sorted(df["main_category"].dropna().unique().tolist())
        selected_cat = st.selectbox("📂 Choose a category", categories)
    with c2:
        sort_options = {
            "Weighted Score (Best Overall)": "weighted_score",
            "Highest Rating": "rating",
            "Biggest Discount": "discount_percentage",
            "Lowest Price": "discounted_price"
        }
        sort_label = st.selectbox("📊 Sort by", list(sort_options.keys()))
        sort_key   = sort_options[sort_label]

    if st.button("🏆 Show Top Picks"):
        recs = get_category_recommendations(df, selected_cat, top_n, sort_key)
        if recs.empty:
            st.warning("No products found in this category.")
        else:
            # Mini bar chart
            fig, ax = plt.subplots(figsize=(9, 3.5))
            colors = ["#FF9900" if i == 0 else "#FFD699" for i in range(len(recs))]
            short_names = [n[:30] + "…" if len(n) > 30 else n for n in recs["product_name"]]
            ax.barh(short_names[::-1], recs[sort_key].values[::-1], color=colors[::-1], edgecolor="white")
            ax.set_xlabel(sort_label); ax.set_title(f"Top {len(recs)} — {selected_cat} ({sort_label})")
            ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown(f"#### 📋 Top {len(recs)} Products — {selected_cat}")
            for rank, (_, row) in enumerate(recs.iterrows(), 1):
                render_product_card(row, rank=rank)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRICE-FILTERED RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">💸 Find Products in Your Budget</div>', unsafe_allow_html=True)

    price_min = float(df["discounted_price"].min())
    price_max = float(df["discounted_price"].max())

    p1, p2, p3 = st.columns(3)
    with p1:
        min_price = st.number_input("Min Price (₹)", min_value=0.0, max_value=price_max,
                                    value=price_min, step=100.0)
    with p2:
        max_price = st.number_input("Max Price (₹)", min_value=0.0, max_value=price_max,
                                    value=min(5000.0, price_max), step=100.0)
    with p3:
        cat_filter = st.selectbox("Category (optional)",
                                  ["All"] + sorted(df["main_category"].dropna().unique().tolist()))

    # Quick budget presets
    st.markdown("**⚡ Quick Presets:**")
    bp1, bp2, bp3, bp4 = st.columns(4)
    preset = None
    if bp1.button("Under ₹500"):   preset = (0, 500)
    if bp2.button("₹500–₹2,000"):  preset = (500, 2000)
    if bp3.button("₹2,000–₹5,000"):preset = (2000, 5000)
    if bp4.button("₹5,000+"):       preset = (5000, price_max)

    if preset:
        min_price, max_price = preset

    if st.button("🔎 Search by Budget"):
        recs = get_price_range_recommendations(df, min_price, max_price, cat_filter, top_n)
        if recs.empty:
            st.warning("No products found in this price range.")
        else:
            st.success(f"Found **{len(recs)}** top-rated products between ₹{min_price:,.0f} – ₹{max_price:,.0f}")
            for rank, (_, row) in enumerate(recs.iterrows(), 1):
                render_product_card(row, rank=rank)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EDA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    ORANGE = "#FF9900"
    DARK   = "#FF6600"
    LIGHT  = "#FFD699"

    # ── Row 1: Product count by category + Rating distribution
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**Products per Category**")
        cat_counts = df["main_category"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(cat_counts.index[::-1], cat_counts.values[::-1], color=ORANGE, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    str(int(bar.get_width())), va="center", fontsize=9)
        ax.set_xlabel("Number of Products"); ax.set_title("Products by Category")
        ax.set_facecolor("#FFF8F0"); fig.patch.set_facecolor("white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r1c2:
        st.markdown("**Rating Distribution**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["rating"].dropna(), bins=20, color=ORANGE, edgecolor="white", alpha=0.9)
        ax.axvline(df["rating"].mean(), color=DARK, linestyle="--", linewidth=2,
                   label=f"Mean: {df['rating'].mean():.2f}")
        ax.set_xlabel("Rating"); ax.set_ylabel("Count"); ax.set_title("Rating Distribution")
        ax.legend(); ax.set_facecolor("#FFF8F0"); fig.patch.set_facecolor("white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Row 2: Discount distribution + Price vs Rating scatter
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("**Discount % Distribution**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["discount_percentage"].dropna(), bins=25, color=DARK, edgecolor="white", alpha=0.88)
        ax.axvline(df["discount_percentage"].mean(), color=ORANGE, linestyle="--", linewidth=2,
                   label=f"Mean: {df['discount_percentage'].mean():.1f}%")
        ax.set_xlabel("Discount %"); ax.set_ylabel("Count"); ax.set_title("Discount Distribution")
        ax.legend(); ax.set_facecolor("#FFF8F0"); fig.patch.set_facecolor("white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r2c2:
        st.markdown("**Price vs Rating**")
        scatter_df = df[df["discounted_price"] < df["discounted_price"].quantile(0.95)].copy()
        fig, ax = plt.subplots(figsize=(7, 4))
        sc = ax.scatter(scatter_df["discounted_price"], scatter_df["rating"],
                        c=scatter_df["discount_percentage"], cmap="YlOrRd",
                        alpha=0.6, edgecolors="none", s=30)
        plt.colorbar(sc, ax=ax, label="Discount %")
        ax.set_xlabel("Discounted Price (₹)"); ax.set_ylabel("Rating")
        ax.set_title("Price vs Rating (colour = Discount %)")
        ax.set_facecolor("#FFF8F0"); fig.patch.set_facecolor("white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Row 3: Avg rating per category + Top 10 most reviewed
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown("**Avg Rating by Category**")
        avg_rat = df.groupby("main_category")["rating"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        colors_list = [ORANGE if v == avg_rat.max() else LIGHT for v in avg_rat.values]
        ax.bar(avg_rat.index, avg_rat.values, color=colors_list, edgecolor="white")
        ax.set_ylabel("Average Rating"); ax.set_title("Average Rating per Category")
        ax.set_ylim(3.5, 5); plt.xticks(rotation=30, ha="right", fontsize=9)
        ax.set_facecolor("#FFF8F0"); fig.patch.set_facecolor("white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r3c2:
        st.markdown("**Top 10 Most Reviewed Products**")
        top_reviewed = df.nlargest(10, "rating_count")[["product_name", "rating_count", "rating"]]
        top_reviewed["short_name"] = top_reviewed["product_name"].str[:35] + "…"
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(top_reviewed["short_name"][::-1], top_reviewed["rating_count"][::-1],
                       color=ORANGE, edgecolor="white")
        ax.set_xlabel("Number of Ratings"); ax.set_title("Top 10 Most Reviewed Products")
        ax.tick_params(axis="y", labelsize=8); ax.set_facecolor("#FFF8F0")
        fig.patch.set_facecolor("white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Row 4: Avg discount by category + Price box by category
    r4c1, r4c2 = st.columns(2)

    with r4c1:
        st.markdown("**Avg Discount % by Category**")
        avg_disc = df.groupby("main_category")["discount_percentage"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(avg_disc.index, avg_disc.values, color=DARK, alpha=0.88, edgecolor="white")
        for i, v in enumerate(avg_disc.values):
            ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
        ax.set_ylabel("Avg Discount %"); ax.set_title("Average Discount % by Category")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        ax.set_facecolor("#FFF8F0"); fig.patch.set_facecolor("white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r4c2:
        st.markdown("**Price Distribution by Category**")
        box_df = df[df["discounted_price"] < df["discounted_price"].quantile(0.95)]
        cats_ordered = box_df.groupby("main_category")["discounted_price"].median().sort_values().index
        data_by_cat = [box_df[box_df["main_category"] == c]["discounted_price"].dropna().values
                       for c in cats_ordered]
        fig, ax = plt.subplots(figsize=(7, 4))
        bp = ax.boxplot(data_by_cat, labels=cats_ordered, patch_artist=True,
                        medianprops=dict(color=DARK, linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(LIGHT)
        ax.set_ylabel("Discounted Price (₹)"); ax.set_title("Price Distribution by Category")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        ax.set_facecolor("#FFF8F0"); fig.patch.set_facecolor("white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Correlation heatmap ──────────────────────────────────────────────────
    st.markdown("**Correlation: Numeric Features**")
    corr_cols = ["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count", "weighted_score"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, annot_kws={"size": 10})
    ax.set_title("Feature Correlation Heatmap")
    fig.patch.set_facecolor("white")
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🗂️ Data Explorer</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    cat_f    = col_a.multiselect("Category", sorted(df["main_category"].unique()),
                                  default=sorted(df["main_category"].unique()))
    min_rat  = col_b.slider("Min Rating", 1.0, 5.0, 3.5, 0.1)
    max_disc = col_c.slider("Min Discount %", 0, 100, 0)

    filtered = df[
        df["main_category"].isin(cat_f) &
        (df["rating"] >= min_rat) &
        (df["discount_percentage"] >= max_disc)
    ]

    st.success(f"Showing **{len(filtered):,}** products")

    display_cols = ["product_name", "main_category", "discounted_price",
                    "actual_price", "discount_percentage", "rating", "rating_count", "weighted_score"]
    st.dataframe(
        filtered[display_cols].rename(columns={
            "product_name": "Product",
            "main_category": "Category",
            "discounted_price": "Price (₹)",
            "actual_price": "MRP (₹)",
            "discount_percentage": "Discount %",
            "rating": "Rating",
            "rating_count": "Reviews",
            "weighted_score": "Score"
        }).reset_index(drop=True),
        use_container_width=True,
        height=450
    )

    st.download_button(
        "⬇️ Download Filtered Data as CSV",
        data=filtered[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="amazon_filtered_products.csv",
        mime="text/csv"
    )

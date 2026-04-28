## Section 14: Interactive Dashboard (Streamlit App)
# Streamlit application for exploring and filtering reviews with real-time insights.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ── page config & helpers ────────────────────────────────────
st.set_page_config(
    page_title="Amazon Reviews Dashboard",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def get_sia():
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon")
        return SentimentIntensityAnalyzer()

SIA = get_sia()


def score_sentiment(text: object) -> Optional[float]:
    if isinstance(text, str) and text.strip():
        return SIA.polarity_scores(text)["compound"]
    return np.nan


def categorize_sentiment(score: object) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def normalize_boolean_series(series: pd.Series) -> pd.Series:
    bool_map = {
        "TRUE": True, "FALSE": False,
        "True": True, "False": False,
        True: True, False: False,
        "yes": True, "no": False,
        "Yes": True, "No": False,
    }
    return series.map(bool_map).fillna(series)


def preprocess_reviews(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    for column in ["reviews.doRecommend", "reviews.didPurchase"]:
        if column in cleaned.columns:
            cleaned[column] = normalize_boolean_series(cleaned[column])
    for column in ["reviews.date", "reviews.dateAdded", "dateAdded", "dateUpdated"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")
    if "reviews.rating" in cleaned.columns:
        cleaned["reviews.rating"] = pd.to_numeric(cleaned["reviews.rating"], errors="coerce")
    for column in ["reviews.text", "reviews.title", "name", "brand", "asins"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()
    if "reviews.text" in cleaned.columns:
        cleaned["sentiment_score"] = cleaned["reviews.text"].apply(score_sentiment)
        cleaned["sentiment_category"] = cleaned["sentiment_score"].apply(categorize_sentiment)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


# ── data loading ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv_from_path(path_text: str) -> pd.DataFrame:
    return pd.read_csv(Path(path_text).expanduser())


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.BytesIO(file_bytes))


def build_dataset_map(uploaded_files, local_paths_text: str) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for uploaded_file in uploaded_files:
        raw = load_uploaded_csv(uploaded_file.getvalue())
        datasets[Path(uploaded_file.name).stem] = preprocess_reviews(raw)
    for raw_path in [l.strip() for l in local_paths_text.splitlines() if l.strip()]:
        path = Path(raw_path).expanduser()
        if path.exists() and path.suffix.lower() == ".csv":
            datasets[path.stem] = preprocess_reviews(load_csv_from_path(raw_path))
    return datasets


def safe_unique_options(series: pd.Series, limit: int = 25) -> List[str]:
    return (
        series.replace("", np.nan)
        .dropna()
        .astype(str)
        .value_counts()
        .head(limit)
        .index.tolist()
    )

# ── Streamlit App – Filters and Metrics ──────────────────────
st.title("Amazon Consumer Reviews Dashboard")
st.caption("Interactive review exploration with dataset selection, rating and sentiment filters, and quick summary charts.")

with st.sidebar:
    st.header("Data Source")
    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )
    local_paths_text = st.text_area(
        "Or paste local CSV paths, one per line",
        value="",
        height=100,
        help="Useful if the CSVs are already on your machine.",
    )

datasets = build_dataset_map(uploaded_files or [], local_paths_text)

if not datasets:
    st.info("Upload at least one CSV file or paste a local path to start the dashboard.")
    st.stop()

dataset_options = sorted(datasets.keys())
dataset_name = st.sidebar.selectbox("Active dataset", options=dataset_options)

if dataset_name is None and dataset_options:
    dataset_name = dataset_options[0]

df = datasets[dataset_name]

st.sidebar.header("Filters")
filtered_df = df.copy()

if "reviews.rating" in filtered_df.columns and filtered_df["reviews.rating"].notna().any():
    rating_min = float(filtered_df["reviews.rating"].dropna().min())
    rating_max = float(filtered_df["reviews.rating"].dropna().max())
    rating_range = st.sidebar.slider(
        "Rating range",
        min_value=rating_min, max_value=rating_max,
        value=(rating_min, rating_max), step=0.5,
    )
    filtered_df = filtered_df[
        filtered_df["reviews.rating"].between(*rating_range, inclusive="both")
        | filtered_df["reviews.rating"].isna()
    ]

if "sentiment_category" in filtered_df.columns:
    sentiment_options = ["Positive", "Neutral", "Negative", "Unknown"]
    sentiment_selected = st.sidebar.multiselect(
        "Sentiment categories",
        options=sentiment_options,
        default=[o for o in sentiment_options if o in filtered_df["sentiment_category"].unique()],
    )
    if sentiment_selected:
        filtered_df = filtered_df[filtered_df["sentiment_category"].isin(sentiment_selected)]

if "brand" in filtered_df.columns:
    brand_options = safe_unique_options(filtered_df["brand"], 20)
    brand_selected = st.sidebar.multiselect("Brand filter", options=brand_options, default=brand_options)
    if brand_selected:
        filtered_df = filtered_df[filtered_df["brand"].astype(str).isin(brand_selected)]

if "name" in filtered_df.columns:
    product_options = safe_unique_options(filtered_df["name"], 20)
    product_selected = st.sidebar.multiselect("Product filter", options=product_options, default=product_options)
    if product_selected:
        filtered_df = filtered_df[filtered_df["name"].astype(str).isin(product_selected)]

if "reviews.text" in filtered_df.columns:
    search_term = st.sidebar.text_input("Search review text", value="").strip()
    if search_term:
        filtered_df = filtered_df[filtered_df["reviews.text"].str.contains(search_term, case=False, na=False)]

if "sentiment_score" in filtered_df.columns:
    sentiment_score_range = st.sidebar.slider(
        "Sentiment score range",
        min_value=-1.0, max_value=1.0,
        value=(-1.0, 1.0), step=0.01,
    )
    filtered_df = filtered_df[
        filtered_df["sentiment_score"].between(*sentiment_score_range, inclusive="both")
        | filtered_df["sentiment_score"].isna()
    ]

top_left, top_mid, top_right, top_far = st.columns(4)

top_left.metric("Filtered reviews", f"{len(filtered_df):,}")
top_mid.metric(
    "Avg rating",
    f"{filtered_df['reviews.rating'].mean():.2f}" if "reviews.rating" in filtered_df.columns else "N/A",
)
top_right.metric(
    "Avg sentiment",
    f"{filtered_df['sentiment_score'].mean():.3f}" if "sentiment_score" in filtered_df.columns else "N/A",
)
if "reviews.doRecommend" in filtered_df.columns:
    recommend_rate = (
        filtered_df["reviews.doRecommend"].astype(str).isin(["True", "TRUE", "1", "Yes", "yes"]).mean() * 100
    )
    top_far.metric("Recommend rate", f"{recommend_rate:.1f}%")
else:
    top_far.metric("Rows after filters", f"{len(filtered_df):,}")

# ── Streamlit App – Charts and Table ─────────────────────────
st.subheader(f"Dataset overview: {dataset_name}")

overview_col_1, overview_col_2 = st.columns(2)

with overview_col_1:
    if "reviews.rating" in filtered_df.columns:
        rating_counts = (
            filtered_df["reviews.rating"].dropna().round().astype(int)
            .value_counts().sort_index().reset_index()
        )
        rating_counts.columns = ["rating", "count"]
        fig = px.bar(rating_counts, x="rating", y="count", title="Rating distribution")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if "sentiment_category" in filtered_df.columns:
        sentiment_counts = (
            filtered_df["sentiment_category"].fillna("Unknown")
            .value_counts().reset_index()
        )
        sentiment_counts.columns = ["sentiment_category", "count"]
        fig = px.bar(
            sentiment_counts, x="sentiment_category", y="count",
            title="VADER sentiment distribution", color="sentiment_category",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with overview_col_2:
    date_columns = [
        c for c in ["reviews.date", "reviews.dateAdded", "dateAdded", "dateUpdated"]
        if c in filtered_df.columns
    ]
    if date_columns:
        date_column = date_columns[0]
        monthly = (
            filtered_df.dropna(subset=[date_column])
            .assign(
                month=lambda frame: pd.to_datetime(
                    frame[date_column], errors="coerce"
                ).dt.to_period("M").astype(str)
            )
            .groupby("month").size().reset_index(name="count")
            .sort_values("month")
        )
        if not monthly.empty:
            fig = px.line(
                monthly, x="month", y="count", markers=True,
                title=f"Review volume by month ({date_column})",
            )
            st.plotly_chart(fig, use_container_width=True)

    if "sentiment_score" in filtered_df.columns and "reviews.rating" in filtered_df.columns:
        box_df = filtered_df[["reviews.rating", "sentiment_score"]].dropna()
        if not box_df.empty:
            fig = px.box(box_df, x="reviews.rating", y="sentiment_score", title="VADER sentiment by rating")
            st.plotly_chart(fig, use_container_width=True)

st.subheader("Top entities")
entity_col_1, entity_col_2 = st.columns(2)

with entity_col_1:
    if "brand" in filtered_df.columns:
        top_brands = (
            filtered_df[filtered_df["brand"].astype(str) != ""]
            .groupby("brand").size().sort_values(ascending=False)
            .head(10).reset_index(name="count")
        )
        if not top_brands.empty:
            fig = px.bar(top_brands, x="count", y="brand", orientation="h", title="Top brands by review count")
            st.plotly_chart(fig, use_container_width=True)

with entity_col_2:
    if "name" in filtered_df.columns:
        top_products = (
            filtered_df[filtered_df["name"].astype(str) != ""]
            .groupby("name").size().sort_values(ascending=False)
            .head(10).reset_index(name="count")
        )
        if not top_products.empty:
            fig = px.bar(top_products, x="count", y="name", orientation="h", title="Top products by review count")
            st.plotly_chart(fig, use_container_width=True)

st.subheader("Filtered records")

preview_columns = [
    c for c in [
        "reviews.title", "reviews.text", "reviews.rating",
        "sentiment_category", "sentiment_score", "brand",
        "name", "reviews.doRecommend", "reviews.date",
    ]
    if c in filtered_df.columns
]

preview_limit = st.slider("Rows to preview", 5, 100, 20)
st.dataframe(
    filtered_df[preview_columns].head(preview_limit) if preview_columns else filtered_df.head(preview_limit),
    use_container_width=True,
)

csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered data as CSV",
    data=csv_bytes,
    file_name=f"{dataset_name}_filtered.csv",
    mime="text/csv",
)

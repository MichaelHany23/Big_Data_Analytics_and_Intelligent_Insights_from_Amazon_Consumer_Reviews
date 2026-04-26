from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


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
        "TRUE": True,
        "FALSE": False,
        "True": True,
        "False": False,
        True: True,
        False: False,
        "yes": True,
        "no": False,
        "Yes": True,
        "No": False,
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

    for column in ["reviews.text", "reviews.title", "name", "brand"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()

    if "reviews.text" in cleaned.columns and "sentiment_score" not in cleaned.columns:
        cleaned["sentiment_score"] = cleaned["reviews.text"].fillna("").apply(score_sentiment)
        cleaned["sentiment_category"] = cleaned["sentiment_score"].apply(categorize_sentiment)
    elif "sentiment_score" in cleaned.columns and "sentiment_category" not in cleaned.columns:
        cleaned["sentiment_category"] = cleaned["sentiment_score"].apply(categorize_sentiment)

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


@st.cache_data(show_spinner=False)
def load_csv_from_path(path_text: str) -> pd.DataFrame:
    path = Path(path_text).expanduser()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.BytesIO(file_bytes))


def build_dataset_map(uploaded_files, local_paths_text: str) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}

    for uploaded_file in uploaded_files:
        raw = load_uploaded_csv(uploaded_file.getvalue())
        datasets[Path(uploaded_file.name).stem] = preprocess_reviews(raw)

    for raw_path in [line.strip() for line in local_paths_text.splitlines() if line.strip()]:
        path = Path(raw_path).expanduser()
        if path.exists() and path.suffix.lower() == ".csv":
            datasets[path.stem] = preprocess_reviews(load_csv_from_path(raw_path))

    return datasets


def safe_unique_options(series: pd.Series, limit: int = 25) -> List[str]:
    values = (
        series.replace("", np.nan)
        .dropna()
        .astype(str)
        .value_counts()
        .head(limit)
        .index.tolist()
    )
    return values


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    if "reviews.rating" in filtered.columns and filtered["reviews.rating"].notna().any():
        rating_min = float(np.nanmin(filtered["reviews.rating"].dropna()))
        rating_max = float(np.nanmax(filtered["reviews.rating"].dropna()))
        rating_range = st.sidebar.slider(
            "Rating range",
            min_value=rating_min,
            max_value=rating_max,
            value=(rating_min, rating_max),
            step=0.5,
        )
        filtered = filtered[
            filtered["reviews.rating"].between(rating_range[0], rating_range[1], inclusive="both")
            | filtered["reviews.rating"].isna()
        ]

    if "sentiment_category" in filtered.columns:
        sentiment_options = ["Positive", "Neutral", "Negative", "Unknown"]
        sentiment_selected = st.sidebar.multiselect(
            "Sentiment categories",
            options=sentiment_options,
            default=[option for option in sentiment_options if option in filtered["sentiment_category"].unique()],
        )
        if sentiment_selected:
            filtered = filtered[filtered["sentiment_category"].isin(sentiment_selected)]

    if "reviews.doRecommend" in filtered.columns:
        recommendation_values = sorted(filtered["reviews.doRecommend"].dropna().astype(str).unique().tolist())
        recommendation_selected = st.sidebar.multiselect(
            "Recommendation status",
            options=recommendation_values,
            default=recommendation_values,
        )
        if recommendation_selected:
            filtered = filtered[filtered["reviews.doRecommend"].astype(str).isin(recommendation_selected)]

    if "brand" in filtered.columns:
        brand_limit = st.sidebar.slider("Top brands to offer", 5, 50, 20)
        brand_options = safe_unique_options(filtered["brand"], brand_limit)
        brand_selected = st.sidebar.multiselect("Brand filter", options=brand_options, default=brand_options)
        if brand_selected:
            filtered = filtered[filtered["brand"].astype(str).isin(brand_selected)]

    if "name" in filtered.columns:
        product_limit = st.sidebar.slider("Top products to offer", 5, 50, 20)
        product_options = safe_unique_options(filtered["name"], product_limit)
        product_selected = st.sidebar.multiselect("Product filter", options=product_options, default=product_options)
        if product_selected:
            filtered = filtered[filtered["name"].astype(str).isin(product_selected)]

    date_columns = [column for column in ["reviews.date", "reviews.dateAdded", "dateAdded", "dateUpdated"] if column in filtered.columns]
    if date_columns:
        date_column = st.sidebar.selectbox("Date field", date_columns)
        date_series = pd.to_datetime(filtered[date_column], errors="coerce")
        valid_dates = date_series.dropna()
        if not valid_dates.empty:
            date_bounds = (valid_dates.min().date(), valid_dates.max().date())
            selected_dates = st.sidebar.date_input("Date range", value=date_bounds)
            if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                start_date, end_date = selected_dates
                mask = date_series.dt.date.between(start_date, end_date)
                filtered = filtered[mask | date_series.isna()]

    if "reviews.text" in filtered.columns:
        search_term = st.sidebar.text_input("Search review text", value="").strip()
        if search_term:
            filtered = filtered[filtered["reviews.text"].str.contains(search_term, case=False, na=False)]

    if "sentiment_score" in filtered.columns:
        sentiment_score_range = st.sidebar.slider(
            "Sentiment score range",
            min_value=-1.0,
            max_value=1.0,
            value=(-1.0, 1.0),
            step=0.01,
        )
        filtered = filtered[
            filtered["sentiment_score"].between(sentiment_score_range[0], sentiment_score_range[1], inclusive="both")
            | filtered["sentiment_score"].isna()
        ]

    if "transformer_sentiment_score" in filtered.columns:
        transformer_range = st.sidebar.slider(
            "Transformer score range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
        )
        filtered = filtered[
            filtered["transformer_sentiment_score"].between(transformer_range[0], transformer_range[1], inclusive="both")
            | filtered["transformer_sentiment_score"].isna()
        ]

    return filtered


def metric_value(series: pd.Series, reducer) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return float("nan")
    return float(reducer(values.dropna()))


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

dataset_name = st.sidebar.selectbox("Active dataset", options=sorted(datasets.keys()))
df = datasets[dataset_name]

st.sidebar.header("Filters")
filtered_df = apply_filters(df)

top_left, top_mid, top_right, top_far = st.columns(4)

top_left.metric("Filtered reviews", f"{len(filtered_df):,}")
top_mid.metric(
    "Avg rating",
    f"{metric_value(filtered_df['reviews.rating'], np.mean):.2f}" if "reviews.rating" in filtered_df.columns else "N/A",
)
top_right.metric(
    "Avg sentiment",
    f"{metric_value(filtered_df['sentiment_score'], np.mean):.3f}" if "sentiment_score" in filtered_df.columns else "N/A",
)
if "reviews.doRecommend" in filtered_df.columns:
    recommend_rate = pd.Series(filtered_df["reviews.doRecommend"].astype(str)).isin(["True", "TRUE", "1", "Yes", "yes"]).mean() * 100
    top_far.metric("Recommend rate", f"{recommend_rate:.1f}%")
elif "transformer_sentiment_label" in filtered_df.columns:
    positive_rate = (filtered_df["transformer_sentiment_label"].astype(str).str.upper() == "POSITIVE").mean() * 100
    top_far.metric("Positive transformer rate", f"{positive_rate:.1f}%")
else:
    top_far.metric("Rows after filters", f"{len(filtered_df):,}")

st.subheader(f"Dataset overview: {dataset_name}")

overview_col_1, overview_col_2 = st.columns(2)

with overview_col_1:
    if "reviews.rating" in filtered_df.columns:
        rating_counts = filtered_df["reviews.rating"].dropna().round().astype(int).value_counts().sort_index().reset_index()
        rating_counts.columns = ["rating", "count"]
        fig = px.bar(rating_counts, x="rating", y="count", title="Rating distribution")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if "sentiment_category" in filtered_df.columns:
        sentiment_counts = filtered_df["sentiment_category"].fillna("Unknown").value_counts().reset_index()
        sentiment_counts.columns = ["sentiment_category", "count"]
        fig = px.bar(sentiment_counts, x="sentiment_category", y="count", title="VADER sentiment distribution", color="sentiment_category")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with overview_col_2:
    date_columns = [column for column in ["reviews.date", "reviews.dateAdded", "dateAdded", "dateUpdated"] if column in filtered_df.columns]
    if date_columns:
        date_column = date_columns[0]
        monthly = (
            filtered_df.dropna(subset=[date_column])
            .assign(month=lambda frame: pd.to_datetime(frame[date_column], errors="coerce").dt.to_period("M").astype(str))
            .groupby("month")
            .size()
            .reset_index(name="count")
        )
        monthly = monthly.sort_values("month")
        if not monthly.empty:
            fig = px.line(monthly, x="month", y="count", markers=True, title=f"Review volume by month ({date_column})")
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
            .groupby("brand")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="count")
        )
        if not top_brands.empty:
            fig = px.bar(top_brands, x="count", y="brand", orientation="h", title="Top brands by review count")
            st.plotly_chart(fig, use_container_width=True)

with entity_col_2:
    if "name" in filtered_df.columns:
        top_products = (
            filtered_df[filtered_df["name"].astype(str) != ""]
            .groupby("name")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="count")
        )
        if not top_products.empty:
            fig = px.bar(top_products, x="count", y="name", orientation="h", title="Top products by review count")
            st.plotly_chart(fig, use_container_width=True)

st.subheader("Filtered records")

preview_columns = [
    column
    for column in [
        "reviews.title",
        "reviews.text",
        "reviews.rating",
        "sentiment_category",
        "sentiment_score",
        "transformer_sentiment_label",
        "transformer_sentiment_score",
        "brand",
        "name",
        "reviews.doRecommend",
        "reviews.date",
    ]
    if column in filtered_df.columns
]

preview_limit = st.slider("Rows to preview", 5, 100, 20)
st.dataframe(filtered_df[preview_columns].head(preview_limit) if preview_columns else filtered_df.head(preview_limit), use_container_width=True)

csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered data as CSV",
    data=csv_bytes,
    file_name=f"{dataset_name}_filtered.csv",
    mime="text/csv",
)

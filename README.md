# Big Data Analytics and Intelligent Insights from Amazon Consumer Reviews

An end-to-end analysis of Amazon product reviews using exploratory data analysis, lexicon-based sentiment analysis, transformer-based sentiment classification, and lightweight machine-learning techniques. The project includes a Jupyter notebook for the full analytical workflow and a Streamlit dashboard for interactive review exploration.

## Project Goals

- Understand rating, recommendation, product, brand, and review-volume patterns.
- Convert unstructured review text into sentiment signals that can be compared with ratings.
- Compare traditional VADER sentiment scoring with a pretrained transformer model.
- Explore review behavior, common complaint or feature terms, and business-oriented actions.
- Provide an interactive interface for filtering reviews and downloading the resulting subset.

## Repository Contents

| File | Description |
| --- | --- |
| `Big_Data_Analytics_and_Intelligent_Insights_from_Amazon_Consumer_Reviews.ipynb` | Main notebook covering ingestion, preprocessing, EDA, sentiment analysis, modeling, behavior analysis, and recommendations. |
| `amazon_reviews_streamlit_app.py` | Interactive Streamlit dashboard for loading, filtering, visualizing, and exporting review data. |
| `1429_1.csv` | Included Amazon review dataset with approximately 35,075 data rows. |
| `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv` | Datafiniti May 2019 review dataset with approximately 11,021 data rows. |
| `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv` | Additional Datafiniti review dataset with approximately 771 data rows. |
| `Idea selection report.pdf` | Project planning and idea-selection reference document. |

## Tools and Technologies

- **Python** for data processing and analysis.
- **Jupyter Notebook** for reproducible, step-by-step exploration.
- **Pandas and NumPy** for tabular data preparation, type conversion, aggregation, and numerical calculations.
- **Matplotlib and Seaborn** for notebook visualizations.
- **Plotly Express and Streamlit** for interactive dashboard charts and controls.
- **NLTK VADER** for fast compound sentiment scoring and Positive/Neutral/Negative categorization.
- **Hugging Face Transformers and PyTorch** for pretrained transformer sentiment inference, including batched CPU/GPU execution.
- **scikit-learn** for TF-IDF text features, train/test splitting, linear regression, and regression metrics.

## Analytical Workflow

1. **Data ingestion:** Load the included CSV files into Pandas DataFrames and register them as separate datasets.
2. **Preprocessing:** Normalize Boolean fields, parse date fields, coerce ratings to numeric values, clean text fields, and remove duplicate rows.
3. **VADER sentiment:** Score review text with NLTK's `SentimentIntensityAnalyzer`. The compound score ranges from `-1` to `1`; scores at or above `0.05` are Positive, scores at or below `-0.05` are Negative, and values between them are Neutral.
4. **Exploratory analysis:** Examine rating distributions, sentiment distributions, review volume over time, sentiment by rating, and the most-reviewed brands and products.
5. **Transformer sentiment:** Run a pretrained text-classification model over review text in batches, then compare transformer labels/scores with ratings and VADER results.
6. **Modeling and behavioral analysis:** Use TF-IDF features for a baseline rating-prediction experiment, inspect user-level review behavior, identify frequent terms, and produce action-oriented recommendation tables.

## Installation

Python 3.9 or newer is recommended. From the project directory, create an environment and install the packages used by the notebook and dashboard:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn plotly streamlit nltk scikit-learn jupyter transformers torch
```

The dashboard downloads the NLTK VADER lexicon automatically on first launch if it is not available. The notebook contains optional installation/download cells for notebook environments such as Google Colab.

## Run the Streamlit Dashboard

```bash
streamlit run amazon_reviews_streamlit_app.py
```

In the sidebar, upload one or more CSV files or paste local CSV paths, then choose the active dataset. The dashboard provides:

- Rating, sentiment, recommendation, brand, product, date, and review-text filters.
- Metrics for filtered review count, average rating, average VADER sentiment, and recommendation rate.
- Rating and sentiment distributions, monthly review volume, and sentiment-by-rating plots.
- Top-brand and top-product views.
- A filtered-record preview and CSV download.

The dashboard is schema-tolerant: optional charts and filters appear only when their expected columns are present. The primary expected fields include `reviews.text`, `reviews.rating`, `reviews.date`, `reviews.doRecommend`, `brand`, and `name`.

## Run the Notebook

Start Jupyter from the project directory:

```bash
jupyter notebook Big_Data_Analytics_and_Intelligent_Insights_from_Amazon_Consumer_Reviews.ipynb
```

Run the cells in order. The Google Drive cells are optional and intended for Colab; when working locally, the included CSV filenames can be used directly. Transformer inference may take substantial time on CPU and may download model files on first use. A CUDA-enabled PyTorch installation can reduce runtime when compatible hardware is available.

## Data Source

The Datafiniti data is available through the Kaggle dataset [Consumer Reviews of Amazon Products](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products). The CSV files are retained in this repository for local execution. Review text, ratings, dates, recommendation flags, product names, and brand fields are used where available.

## Important Notes and Limitations

- Sentiment is inferred from review text and should not be treated as a ground-truth label.
- VADER is a general-purpose lexicon model and can miss product-specific context, sarcasm, or mixed opinions.
- Transformer inference is computationally expensive and model outputs depend on the selected pretrained model and preprocessing choices.
- Missing dates, ratings, text, or metadata are preserved where practical, but some charts cannot be rendered without the relevant columns.
- Dataset snapshots may contain overlapping records or different schemas; the notebook and dashboard process each file independently.
- The Streamlit app does not run transformer inference itself. If a CSV includes `transformer_sentiment_label` and `transformer_sentiment_score`, it can filter and display those precomputed values; transformer generation is performed in the notebook workflow.
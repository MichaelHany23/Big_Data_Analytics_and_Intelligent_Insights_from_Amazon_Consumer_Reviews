# Big Data Analytics and Intelligent Insights from Amazon Consumer Reviews

## Overview

This project provides a comprehensive data analysis and interactive visualization dashboard for Amazon Consumer Reviews. It leverages natural language processing (NLP) to perform sentiment analysis and extracts actionable insights from product reviews. 

The project includes an extensive Jupyter Notebook for exploratory data analysis (EDA) and model building, as well as an interactive Streamlit web application.

## Key Features

- **Exploratory Data Analysis (EDA):** Deep dive into review scores, purchase patterns, and trends over time.
- **Sentiment Analysis:** Utilizes NLTK's VADER SentimentIntensityAnalyzer to classify review texts into positive, negative, and neutral sentiments.
- **Interactive Dashboard:** A Streamlit app that provides real-time visualizations (powered by Plotly) and filtering mechanisms to explore the dataset dynamically.
- **Data Preprocessing:** Robust extraction, cleaning, and normalization of text and boolean features.

## Project Structure

- `Amazon_Reviews_Comprehensive_Analysis.ipynb`: The main Jupyter Notebook detailing the step-by-step data analytics, NLP sentiment scoring, machine learning operations, and visualization.
- `streamlit_app.py`: The code for the interactive Streamlit dashboard.
- Datasets (CSV files): Contains the raw and structured data utilized for the analysis, such as `1429_1.csv` and `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`.
- `exports/`, `models/`, `plots/`: Directories designated for outputting analysis results, trained models, and generated figures.

## How to Run the Dashboard

1. **Install Dependencies:**
   Ensure you have the required Python packages installed. You may need:
   ```bash
   pip install pandas numpy streamlit plotly nltk jupyter
   ```

2. **Launch Streamlit App:**
   From the root directory of the project, run the following command in your terminal:
   ```bash
   streamlit run streamlit_app.py
   ```
   
3. **Explore Jupyter Notebooks:**
   Start the Jupyter server to dive into the comprehensive data analysis steps:
   ```bash
   jupyter notebook Amazon_Reviews_Comprehensive_Analysis.ipynb
   ```

## Requirements

- Python 3.7+
- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `nltk`
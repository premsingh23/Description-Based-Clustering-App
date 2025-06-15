# Description-Based Clustering App

This prototype demonstrates a simple Streamlit application for clustering archaeological text descriptions using OpenAI embeddings.

## Features

- Load your own CSV file or try the bundled example dataset.
- Basic text preprocessing with NLTK.
- Generate embeddings via the OpenAI API.
- K-Means clustering with adjustable number of clusters.
- Simple visualizations of clusters and dimension importance.

## Setup

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Obtain an OpenAI API key from <https://platform.openai.com> and keep it handy.

## Running Locally

```bash
streamlit run app.py
```

The app will prompt for your API key when generating embeddings.

## Streamlit Cloud

Deploy the repository on [Streamlit Community Cloud](https://streamlit.io/cloud) and set the `OPENAI_API_KEY` secret in the deployment settings.

## Example Data

A small archaeology-themed dataset is provided in `examples/archaeology_samples.csv` for quick testing.

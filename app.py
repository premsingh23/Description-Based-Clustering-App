import streamlit as st
import pandas as pd
from modules import data_loader, preprocessor, embeddings, dimension_analysis, weighting, clustering, visualization
from modules.ai_assistant import AssistantAgent

st.set_page_config(page_title="Description Clustering")

agent = AssistantAgent()
st.sidebar.write(agent.greet())

if "data" not in st.session_state:
    st.session_state.data = None

st.title("Description-Based Clustering App")

# Data selection
st.header("1. Load Data")
upload = st.file_uploader("Upload CSV", type="csv")
use_example = st.checkbox("Use example dataset", value=not upload)
if use_example:
    if st.button("Load Example Data"):
        st.session_state.data = data_loader.load_example()
elif upload is not None:
    st.session_state.data = data_loader.load_csv(upload)

if st.session_state.data is not None:
    df = st.session_state.data.copy()
    st.write(df.head())
    label_col = st.selectbox("Label column", df.columns)
    text_col = st.selectbox("Description column", df.columns)

    if st.button("Preprocess Text"):
        preprocessor.preprocess_dataframe(df, text_col)
        st.session_state.df = df

if "df" in st.session_state:
    st.header("2. Generate Embeddings")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key and st.button("Create Embeddings"):
        texts = st.session_state.df[text_col].tolist()
        st.session_state.embeddings = embeddings.embed_texts(texts, api_key)

if "embeddings" in st.session_state:
    st.header("3. Dimension Analysis")
    if st.button("Analyze Dimensions"):
        imp = dimension_analysis.compute_dimension_variance(st.session_state.df, st.session_state.embeddings, label_col)
        st.session_state.importance = imp
        st.plotly_chart(visualization.importance_bar(imp))

if "embeddings" in st.session_state:
    st.header("4. Clustering")
    k = st.number_input("Number of clusters", min_value=2, max_value=10, value=3)
    if st.button("Run Clustering"):
        labels = clustering.kmeans_cluster(st.session_state.embeddings, int(k))
        st.session_state.df["cluster"] = labels
        st.session_state.labels = labels
        st.plotly_chart(visualization.cluster_scatter(st.session_state.embeddings, labels, st.session_state.df, label_col))

if "labels" in st.session_state:
    st.header("5. Cluster Summary")
    st.write(st.session_state.df.groupby("cluster")[label_col].value_counts())

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import uuid
import string
import io
import base64
from datetime import datetime

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Define page order for navigation
PAGE_ORDER = [
    "Data Upload", 
    "Preprocessing", 
    "Embedding Generation", 
    "Dimension Analysis", 
    "Dimension Weighting", 
    "Clustering", 
    "Visualization & Analysis"
]

# Initialize session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = None
if 'embedding_model_info' not in st.session_state:
    st.session_state['embedding_model_info'] = None
if 'dimension_importance' not in st.session_state:
    st.session_state['dimension_importance'] = None
if 'weighted_embeddings' not in st.session_state:
    st.session_state['weighted_embeddings'] = None
if 'clusters' not in st.session_state:
    st.session_state['clusters'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = PAGE_ORDER[0]

# Navigation functions
def go_to_next_page():
    current_index = PAGE_ORDER.index(st.session_state['current_page'])
    if current_index < len(PAGE_ORDER) - 1:
        st.session_state['current_page'] = PAGE_ORDER[current_index + 1]
    st.rerun()

# Text preprocessing functions
def preprocess_text(text, options):
    if not isinstance(text, str):
        return ""
    
    # Apply selected preprocessing steps
    if options.get('lowercase', False):
        text = text.lower()
    
    if options.get('remove_punctuation', False):
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    if options.get('remove_numbers', False):
        text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if options.get('remove_stopwords', False):
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    if options.get('lemmatize', False):
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

# Embedding generation functions
def generate_embeddings(df, text_column, method="TF-IDF"):
    if method == "TF-IDF":
        vectorizer = TfidfVectorizer(max_features=100)
        embeddings = vectorizer.fit_transform(df[text_column]).toarray()
        
        model_info = {
            "method": "TF-IDF",
            "dimensions": embeddings.shape[1],
            "vocabulary_size": len(vectorizer.vocabulary_)
        }
        
        return embeddings, model_info
    
    elif method == "Bag of Words":
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(max_features=100)
        embeddings = vectorizer.fit_transform(df[text_column]).toarray()
        
        model_info = {
            "method": "Bag of Words",
            "dimensions": embeddings.shape[1],
            "vocabulary_size": len(vectorizer.vocabulary_)
        }
        
        return embeddings, model_info
    
    else:
        # Fallback to TF-IDF if requested method not available
        st.warning(f"{method} is not available. Using TF-IDF instead.")
        return generate_embeddings(df, text_column, "TF-IDF")

# Dimension analysis functions
def analyze_dimensions(df, embeddings, label_column, ref_label_1, ref_label_2):
    # Get indices for each reference label
    indices_1 = df[df[label_column] == ref_label_1].index
    indices_2 = df[df[label_column] == ref_label_2].index
    
    # Get embeddings for each reference label
    embeddings_1 = embeddings[indices_1]
    embeddings_2 = embeddings[indices_2]
    
    # Calculate mean difference for each dimension
    mean_1 = np.mean(embeddings_1, axis=0)
    mean_2 = np.mean(embeddings_2, axis=0)
    mean_diff = np.abs(mean_1 - mean_2)
    
    # Calculate standard deviations
    std_1 = np.std(embeddings_1, axis=0)
    std_2 = np.std(embeddings_2, axis=0)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((std_1**2 + std_2**2) / 2)
    effect_size = mean_diff / (pooled_std + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Calculate p-values using t-test
    from scipy import stats
    _, p_values = stats.ttest_ind(embeddings_1, embeddings_2, axis=0, equal_var=False)
    
    # Create dimension importance
    dimension_importance = []
    for dim in range(len(mean_diff)):
        dimension_importance.append({
            'dimension': dim,
            'importance': mean_diff[dim],
            'effect_size': effect_size[dim],
            'p_value': p_values[dim],
            'mean_1': mean_1[dim],
            'mean_2': mean_2[dim],
            'std_1': std_1[dim],
            'std_2': std_2[dim]
        })
    
    # Sort by importance
    dimension_importance = sorted(dimension_importance, key=lambda x: x['importance'], reverse=True)
    
    return dimension_importance

# Weighting functions
def apply_weights(embeddings, dimension_importance, scheme="linear", top_n=20, weight_factor=2.0):
    # Create weight vector (default weights of 1.0)
    weights = np.ones(embeddings.shape[1])
    
    # Get top N dimensions
    top_dimensions = [item['dimension'] for item in dimension_importance[:top_n]]
    
    # Apply weighting scheme
    if scheme == "linear":
        for i, dim in enumerate(top_dimensions):
            weight = 1.0 + (weight_factor * (1.0 - i / len(top_dimensions)))
            weights[dim] = weight
    
    elif scheme == "exponential":
        for i, dim in enumerate(top_dimensions):
            weight = 1.0 + (weight_factor * np.exp(-i / (len(top_dimensions) / 3)))
            weights[dim] = weight
    
    elif scheme == "binary":
        for dim in top_dimensions:
            weights[dim] = weight_factor
    
    elif scheme == "sigmoid":
        import math
        middle_point = len(top_dimensions) / 2
        for i, dim in enumerate(top_dimensions):
            x = (middle_point - i) / middle_point * 5  # Scale to reasonable sigmoid input
            sigmoid_value = 1 / (1 + math.exp(-x))
            weights[dim] = 1.0 + (weight_factor - 1.0) * sigmoid_value
    
    # Apply weights to embeddings
    weighted_embeddings = embeddings * weights
    
    # Normalize to preserve distances
    scaler = StandardScaler()
    weighted_embeddings = scaler.fit_transform(weighted_embeddings)
    
    return weighted_embeddings

# Clustering functions
def cluster_data(embeddings, algorithm="K-Means", params=None):
    if params is None:
        params = {}
        
    default_params = {
        "K-Means": {"n_clusters": 5, "random_state": 42},
        "Hierarchical": {"n_clusters": 5, "affinity": "euclidean", "linkage": "ward"},
        "DBSCAN": {"eps": 0.5, "min_samples": 5}
    }
    
    # Merge default params with provided params
    for key, value in default_params.get(algorithm, {}).items():
        if key not in params:
            params[key] = value
    
    # Apply clustering algorithm
    if algorithm == "K-Means":
        clusterer = KMeans(
            n_clusters=params["n_clusters"],
            random_state=params["random_state"]
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        
    elif algorithm == "Hierarchical":
        clusterer = AgglomerativeClustering(
            n_clusters=params["n_clusters"],
            affinity=params["affinity"],
            linkage=params["linkage"]
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        
    elif algorithm == "DBSCAN":
        clusterer = DBSCAN(
            eps=params["eps"],
            min_samples=params["min_samples"]
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        
    else:
        st.error(f"Clustering algorithm {algorithm} not available.")
        return None, None
    
    # Calculate clustering metrics
    metrics = {}
    
    # Silhouette score (only if more than one cluster and not all points are in the same cluster)
    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) > 1 and -1 not in unique_labels:
        metrics["silhouette_score"] = silhouette_score(embeddings, cluster_labels)
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(embeddings, cluster_labels)
        metrics["davies_bouldin_score"] = davies_bouldin_score(embeddings, cluster_labels)
    
    return cluster_labels, metrics

# Visualization functions
def reduce_dimensions(embeddings, method="PCA", n_components=2):
    if method == "PCA":
        reducer = PCA(n_components=n_components)
    elif method == "t-SNE":
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method == "TruncatedSVD":
        reducer = TruncatedSVD(n_components=n_components)
    else:
        st.warning(f"{method} not available. Using PCA instead.")
        return reduce_dimensions(embeddings, "PCA", n_components)
    
    return reducer.fit_transform(embeddings)

def get_downloadable_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return href

def main():
    # Main title
    st.title("Description Based Clustering App")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", PAGE_ORDER, index=PAGE_ORDER.index(st.session_state['current_page']))
    
    # Update current page if changed from sidebar
    if selected_page != st.session_state['current_page']:
        st.session_state['current_page'] = selected_page
        st.rerun()
    
    # Display info message about current page
    st.info(f"You are on the {st.session_state['current_page']} page.")

    # Page-specific content
    if st.session_state['current_page'] == "Data Upload":
        st.header("Data Upload")
        st.write("Upload your CSV or Excel file containing labels and text descriptions.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                    
                st.session_state['data'] = df
                st.success(f"File uploaded successfully! {df.shape[0]} rows and {df.shape[1]} columns.")
                
                st.subheader("Preview")
                st.dataframe(df.head())
                
                st.subheader("Column Selection")
                label_col = st.selectbox("Select label column", df.columns)
                desc_col = st.selectbox("Select description column", df.columns)
                
                if st.button("Confirm Selection and Continue"):
                    st.session_state['label_column'] = label_col
                    st.session_state['description_column'] = desc_col
                    st.success(f"Selected '{label_col}' as label column and '{desc_col}' as description column.")
                    go_to_next_page()
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif st.session_state['current_page'] == "Preprocessing":
        st.header("Preprocessing")
        if st.session_state['data'] is None:
            st.warning("Please upload data first.")
            if st.button("Go to Data Upload"):
                st.session_state['current_page'] = "Data Upload"
                st.rerun()
        else:
            df = st.session_state['data']
            desc_col = st.session_state['description_column']
            
            st.write("Apply preprocessing options to your text data.")
            
            # Preprocessing options
            preprocessing_options = {
                'lowercase': st.checkbox("Convert to lowercase", value=True),
                'remove_punctuation': st.checkbox("Remove punctuation", value=True),
                'remove_numbers': st.checkbox("Remove numbers", value=False),
                'remove_stopwords': st.checkbox("Remove stopwords", value=True),
                'lemmatize': st.checkbox("Lemmatize", value=False)
            }
            
            # Sample text
            st.subheader("Sample Text")
            sample_idx = min(0, len(df) - 1)
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Original Text")
                st.text_area("", df.iloc[sample_idx][desc_col], height=150, disabled=True)
            
            with col2:
                st.write("Processed Text")
                processed_text = preprocess_text(df.iloc[sample_idx][desc_col], preprocessing_options)
                st.text_area("", processed_text, height=150, disabled=True)
            
            if st.button("Apply Preprocessing and Continue"):
                with st.spinner("Preprocessing text..."):
                    # Apply preprocessing to all text
                    processed_texts = []
                    for text in df[desc_col]:
                        processed_texts.append(preprocess_text(text, preprocessing_options))
                    
                    # Create new dataframe with processed text
                    df_processed = df.copy()
                    df_processed['processed_text'] = processed_texts
                    
                    # Store in session state
                    st.session_state['processed_data'] = df_processed
                    st.session_state['preprocessing_options'] = preprocessing_options
                    
                    st.success("Preprocessing complete!")
                    go_to_next_page()
    
    elif st.session_state['current_page'] == "Embedding Generation":
        st.header("Embedding Generation")
        if st.session_state['processed_data'] is None:
            st.warning("Please preprocess your data first.")
            if st.button("Go to Preprocessing"):
                st.session_state['current_page'] = "Preprocessing"
                st.rerun()
        else:
            df_processed = st.session_state['processed_data']
            
            st.write("Generate embeddings using your selected embedding model.")
            
            embedding_methods = {
                "TF-IDF": "Term Frequency-Inverse Document Frequency",
                "Bag of Words": "Simple word count vectors"
            }
            
            selected_method = st.selectbox(
                "Select embedding method", 
                list(embedding_methods.keys()),
                format_func=lambda x: f"{x} - {embedding_methods[x]}"
            )
            
            if st.button("Generate Embeddings and Continue"):
                with st.spinner(f"Generating embeddings using {selected_method}..."):
                    # Generate embeddings
                    embeddings, model_info = generate_embeddings(
                        df_processed, 
                        'processed_text', 
                        method=selected_method
                    )
                    
                    # Store in session state
                    st.session_state['embeddings'] = embeddings
                    st.session_state['embedding_model_info'] = model_info
                    
                    # Display embedding information
                    st.success(f"Embeddings generated successfully with {model_info['dimensions']} dimensions!")
                    
                    # Visualize embeddings
                    if embeddings.shape[1] > 2:
                        reduced_embeddings = reduce_dimensions(embeddings, "PCA", 2)
                        
                        fig = px.scatter(
                            x=reduced_embeddings[:, 0],
                            y=reduced_embeddings[:, 1],
                            hover_name=df_processed[st.session_state['label_column']],
                            title="2D PCA visualization of embeddings"
                        )
                        st.plotly_chart(fig)
                    
                    go_to_next_page()
    
    elif st.session_state['current_page'] == "Dimension Analysis":
        st.header("Dimension Analysis")
        if st.session_state['embeddings'] is None:
            st.warning("Please generate embeddings first.")
            if st.button("Go to Embedding Generation"):
                st.session_state['current_page'] = "Embedding Generation"
                st.rerun()
        else:
            df = st.session_state['processed_data']
            label_col = st.session_state['label_column']
            embeddings = st.session_state['embeddings']
            
            st.write("Analyze embedding dimensions to identify which are most important for distinguishing between labels.")
            
            # Get label statistics
            label_counts = df[label_col].value_counts()
            st.subheader("Label Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            label_counts.head(10).plot(kind='bar', ax=ax)
            st.pyplot(fig)
            
            # Select reference labels
            st.subheader("Select Reference Labels")
            top_labels = label_counts.head(10).index.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                ref_label_1 = st.selectbox("Reference Label 1", top_labels, index=0)
            
            with col2:
                remaining_labels = [label for label in top_labels if label != ref_label_1]
                ref_label_2 = st.selectbox("Reference Label 2", remaining_labels, index=0)
            
            if st.button("Analyze Dimensions and Continue"):
                with st.spinner("Analyzing dimensions..."):
                    # Analyze dimensions
                    dimension_importance = analyze_dimensions(
                        df, embeddings, label_col, ref_label_1, ref_label_2
                    )
                    
                    # Store in session state
                    st.session_state['dimension_importance'] = dimension_importance
                    st.session_state['reference_labels'] = (ref_label_1, ref_label_2)
                    
                    # Show top important dimensions
                    st.subheader("Top Important Dimensions")
                    
                    # Plot dimension importance
                    top_n = min(10, len(dimension_importance))
                    importance_df = pd.DataFrame([
                        {
                            'Dimension': d['dimension'],
                            'Importance': d['importance'],
                            'Effect Size': d['effect_size']
                        } for d in dimension_importance[:top_n]
                    ])
                    
                    fig = px.bar(
                        importance_df,
                        x='Dimension',
                        y='Importance',
                        title=f"Top {top_n} Important Dimensions for {ref_label_1} vs {ref_label_2}"
                    )
                    st.plotly_chart(fig)
                    
                    st.success("Dimension analysis complete!")
                    go_to_next_page()
    
    elif st.session_state['current_page'] == "Dimension Weighting":
        st.header("Dimension Weighting")
        if st.session_state['dimension_importance'] is None:
            st.warning("Please complete dimension analysis first.")
            if st.button("Go to Dimension Analysis"):
                st.session_state['current_page'] = "Dimension Analysis"
                st.rerun()
        else:
            embeddings = st.session_state['embeddings']
            dimension_importance = st.session_state['dimension_importance']
            ref_labels = st.session_state['reference_labels']
            
            st.write("Apply weighting schemes to emphasize important dimensions.")
            
            weighting_schemes = {
                "linear": "Linear decay from top to bottom dimensions",
                "exponential": "Exponential decay from top to bottom",
                "binary": "Equal weight to all top dimensions",
                "sigmoid": "Sigmoid weighting with sharp transition"
            }
            
            st.subheader("Weighting Parameters")
            
            selected_scheme = st.selectbox(
                "Select weighting scheme", 
                list(weighting_schemes.keys()),
                format_func=lambda x: f"{x.title()} - {weighting_schemes[x]}"
            )
            
            top_n = st.slider(
                "Number of top dimensions to emphasize", 
                min_value=1, 
                max_value=min(50, len(dimension_importance)),
                value=min(20, len(dimension_importance))
            )
            
            weight_factor = st.slider(
                "Weight factor", 
                min_value=1.0, 
                max_value=10.0, 
                value=2.0, 
                step=0.1
            )
            
            if st.button("Apply Weighting and Continue"):
                with st.spinner(f"Applying {selected_scheme} weighting..."):
                    # Apply weights
                    weighted_embeddings = apply_weights(
                        embeddings,
                        dimension_importance,
                        scheme=selected_scheme,
                        top_n=top_n,
                        weight_factor=weight_factor
                    )
                    
                    # Store in session state
                    st.session_state['weighted_embeddings'] = weighted_embeddings
                    st.session_state['weighting_info'] = {
                        'scheme': selected_scheme,
                        'top_n': top_n,
                        'weight_factor': weight_factor
                    }
                    
                    # Compare original vs weighted embeddings
                    st.subheader("Original vs. Weighted Embeddings")
                    
                    col1, col2 = st.columns(2)
                    
                    # Reduce dimensions for visualization
                    reduced_orig = reduce_dimensions(embeddings, "PCA", 2)
                    reduced_weighted = reduce_dimensions(weighted_embeddings, "PCA", 2)
                    
                    df = st.session_state['processed_data']
                    label_col = st.session_state['label_column']
                    
                    with col1:
                        st.write("Original Embeddings")
                        fig1 = px.scatter(
                            x=reduced_orig[:, 0],
                            y=reduced_orig[:, 1],
                            color=df[label_col],
                            title="Original Embeddings"
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        st.write("Weighted Embeddings")
                        fig2 = px.scatter(
                            x=reduced_weighted[:, 0],
                            y=reduced_weighted[:, 1],
                            color=df[label_col],
                            title="Weighted Embeddings"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.success("Weighting applied successfully!")
                    go_to_next_page()
    
    elif st.session_state['current_page'] == "Clustering":
        st.header("Clustering")
        if st.session_state['weighted_embeddings'] is None:
            st.warning("Please complete dimension weighting first.")
            if st.button("Go to Dimension Weighting"):
                st.session_state['current_page'] = "Dimension Weighting"
                st.rerun()
        else:
            embeddings = st.session_state['weighted_embeddings']
            
            st.write("Cluster your weighted embeddings to discover patterns.")
            
            clustering_algorithms = {
                "K-Means": "Simple and fast clustering based on centroids",
                "Hierarchical": "Hierarchical clustering with customizable linkage",
                "DBSCAN": "Density-based clustering that can find clusters of arbitrary shape"
            }
            
            selected_algorithm = st.selectbox(
                "Select clustering algorithm", 
                list(clustering_algorithms.keys()),
                format_func=lambda x: f"{x} - {clustering_algorithms[x]}"
            )
            
            st.subheader("Algorithm Parameters")
            
            params = {}
            
            if selected_algorithm == "K-Means":
                params['n_clusters'] = st.slider(
                    "Number of clusters (k)", 
                    min_value=2, 
                    max_value=20, 
                    value=5
                )
                params['random_state'] = 42
                
            elif selected_algorithm == "Hierarchical":
                params['n_clusters'] = st.slider(
                    "Number of clusters", 
                    min_value=2, 
                    max_value=20, 
                    value=5
                )
                params['affinity'] = st.selectbox(
                    "Affinity", 
                    ["euclidean", "l1", "l2", "manhattan", "cosine"]
                )
                params['linkage'] = st.selectbox(
                    "Linkage", 
                    ["ward", "complete", "average", "single"]
                )
                
            elif selected_algorithm == "DBSCAN":
                params['eps'] = st.slider(
                    "Epsilon (neighborhood distance)", 
                    min_value=0.1, 
                    max_value=2.0, 
                    value=0.5, 
                    step=0.1
                )
                params['min_samples'] = st.slider(
                    "Minimum samples in neighborhood", 
                    min_value=2, 
                    max_value=20, 
                    value=5
                )
            
            if st.button("Run Clustering and Continue"):
                with st.spinner(f"Running {selected_algorithm} clustering..."):
                    # Perform clustering
                    cluster_labels, metrics = cluster_data(
                        embeddings,
                        algorithm=selected_algorithm,
                        params=params
                    )
                    
                    if cluster_labels is None:
                        st.error("Clustering failed. Please try with different parameters.")
                    else:
                        # Store in session state
                        st.session_state['clusters'] = {
                            'labels': cluster_labels,
                            'algorithm': selected_algorithm,
                            'params': params,
                            'metrics': metrics
                        }
                        
                        # Add cluster labels to dataframe
                        df = st.session_state['processed_data'].copy()
                        df['cluster'] = cluster_labels
                        st.session_state['clustered_data'] = df
                        
                        # Show clustering metrics
                        st.subheader("Clustering Metrics")
                        for metric, value in metrics.items():
                            st.write(f"{metric}: {value:.4f}")
                        
                        # Show cluster distribution
                        st.subheader("Cluster Distribution")
                        
                        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                        fig = px.bar(
                            x=cluster_counts.index.astype(str),
                            y=cluster_counts.values,
                            labels={'x': 'Cluster', 'y': 'Count'},
                            title="Number of Items per Cluster"
                        )
                        st.plotly_chart(fig)
                        
                        st.success(f"Clustering complete! Found {len(np.unique(cluster_labels))} clusters.")
                        go_to_next_page()
    
    elif st.session_state['current_page'] == "Visualization & Analysis":
        st.header("Visualization & Analysis")
        if 'clusters' not in st.session_state or st.session_state['clusters'] is None:
            st.warning("Please complete clustering first.")
            if st.button("Go to Clustering"):
                st.session_state['current_page'] = "Clustering"
                st.rerun()
        else:
            df = st.session_state['clustered_data']
            label_col = st.session_state['label_column']
            desc_col = st.session_state['description_column']
            embeddings = st.session_state['weighted_embeddings']
            clusters = st.session_state['clusters']['labels']
            
            st.write("Visualize and analyze your clusters.")
            
            tabs = st.tabs(["Cluster Visualization", "Label Distribution", "Cluster Analysis"])
            
            with tabs[0]:
                st.subheader("Cluster Visualization")
                
                viz_method = st.selectbox(
                    "Dimension Reduction Method", 
                    ["PCA", "t-SNE"],
                    index=0
                )
                
                viz_dims = st.radio("Dimensions", [2, 3], horizontal=True)
                
                with st.spinner(f"Generating {viz_dims}D visualization using {viz_method}..."):
                    # Reduce dimensions for visualization
                    reduced = reduce_dimensions(embeddings, viz_method, viz_dims)
                    
                    if viz_dims == 2:
                        fig = px.scatter(
                            x=reduced[:, 0],
                            y=reduced[:, 1],
                            color=clusters.astype(str),
                            hover_name=df[label_col],
                            title=f"2D {viz_method} visualization of clusters"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.scatter_3d(
                            x=reduced[:, 0],
                            y=reduced[:, 1],
                            z=reduced[:, 2],
                            color=clusters.astype(str),
                            hover_name=df[label_col],
                            title=f"3D {viz_method} visualization of clusters"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tabs[1]:
                st.subheader("Label Distribution Across Clusters")
                
                # Get top labels
                top_n_labels = st.slider("Number of top labels to show", 5, 20, 10)
                top_labels = df[label_col].value_counts().head(top_n_labels).index
                
                # Filter data for top labels
                filtered_data = df[df[label_col].isin(top_labels)]
                
                # Create cross-tabulation
                crosstab = pd.crosstab(
                    filtered_data[label_col], 
                    filtered_data['cluster'], 
                    normalize='index'
                )
                
                # Plot heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(crosstab, annot=True, cmap="YlGnBu", ax=ax)
                ax.set_title("Label Distribution Across Clusters (Normalized)")
                ax.set_xlabel("Cluster")
                ax.set_ylabel("Label")
                st.pyplot(fig)
            
            with tabs[2]:
                st.subheader("Cluster Analysis")
                
                # Select cluster to analyze
                unique_clusters = sorted(df['cluster'].unique())
                selected_cluster = st.selectbox("Select Cluster to Analyze", unique_clusters)
                
                # Filter data for selected cluster
                cluster_subset = df[df['cluster'] == selected_cluster]
                st.write(f"**Cluster {selected_cluster} contains {len(cluster_subset)} items**")
                
                # Top labels in cluster
                st.write("**Top Labels in Cluster:**")
                top_labels = cluster_subset[label_col].value_counts().head(10)
                fig = px.bar(
                    x=top_labels.index, 
                    y=top_labels.values,
                    labels={'x': label_col, 'y': 'Count'},
                    title=f"Top Labels in Cluster {selected_cluster}"
                )
                st.plotly_chart(fig)
                
                # Sample items from cluster
                st.write("**Sample Items from Cluster:**")
                st.dataframe(cluster_subset[[label_col, desc_col]].head(5))
            
            # Export options
            st.subheader("Export Results")
            export_format = st.radio("Select Export Format", options=["CSV", "Excel"], index=0)
            export_filename = st.text_input("Enter export filename", "results")
            if export_format == "CSV":
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'data:file/csv;base64,{b64}'
                st.markdown(f'<a href="{href}" download="{export_filename}.csv">Download CSV</a>', unsafe_allow_html=True)
            elif export_format == "Excel":
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Results', index=False)
                b64 = base64.b64encode(output.getvalue()).decode()
                href = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
                st.markdown(f'<a href="{href}" download="{export_filename}.xlsx">Download Excel</a>', unsafe_allow_html=True)
            
            st.success(f"Results ready for download!")
    
    # Footer
    st.markdown("---")
    st.markdown("Description-Based Clustering App | Created with Streamlit")

if __name__ == "__main__":
    main() 
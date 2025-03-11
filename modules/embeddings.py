import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import os
import pickle
from pathlib import Path
import hashlib

# Import embedding models conditionally to handle missing dependencies gracefully
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import tensorflow_hub as hub
    TENSORFLOW_HUB_AVAILABLE = True
except ImportError:
    TENSORFLOW_HUB_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def get_available_models() -> Dict[str, Dict]:
    """
    Get available embedding models with descriptions.
    
    Returns:
        Dictionary of available models with metadata
    """
    available_models = {}
    
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        available_models["SentenceBERT"] = {
            "description": "Sentence-BERT models for generating sentence embeddings",
            "dimensions": 384,  # Default for all-MiniLM-L6-v2
            "local": True,
            "requires_api_key": False
        }
    
    if TENSORFLOW_HUB_AVAILABLE:
        available_models["Universal Sentence Encoder"] = {
            "description": "Google's Universal Sentence Encoder for text embeddings",
            "dimensions": 512,
            "local": True,
            "requires_api_key": False
        }
    
    if OPENAI_AVAILABLE:
        available_models["OpenAI"] = {
            "description": "OpenAI's text embedding models (requires API key)",
            "dimensions": 1536,  # For text-embedding-3-small
            "local": False,
            "requires_api_key": True
        }
    
    # Always include custom model option
    available_models["Custom Embedding CSV"] = {
        "description": "Upload pre-computed embeddings from CSV file",
        "dimensions": "Variable",
        "local": True,
        "requires_api_key": False
    }
    
    return available_models


def generate_embeddings(
    df: pd.DataFrame, 
    text_column: str,
    model_name: str = "SentenceBERT",
    model_params: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Generate embeddings for text using the specified model.
    
    Args:
        df: DataFrame containing the text data
        text_column: Column name containing text to embed
        model_name: Name of the embedding model to use
        model_params: Optional parameters for the embedding model
        use_cache: Whether to use cached embeddings if available
        
    Returns:
        Tuple containing:
        - List of embedding vectors
        - Dictionary with model information
    """
    if model_params is None:
        model_params = {}
    
    # Get texts to embed
    texts = df[text_column].tolist()
    
    # Create cache directory if it doesn't exist
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Generate cache key based on texts and model
    cache_key = f"{model_name}_{hashlib.md5(''.join(texts).encode()).hexdigest()}"
    cache_file = cache_dir / f"{cache_key}.pkl"
    
    # Check if cached embeddings exist
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                print(f"Loaded embeddings from cache: {cache_file}")
                return cached_data["embeddings"], cached_data["model_info"]
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    # Generate embeddings based on selected model
    if model_name == "SentenceBERT":
        embeddings, model_info = _generate_sbert_embeddings(texts, model_params)
    
    elif model_name == "Universal Sentence Encoder":
        embeddings, model_info = _generate_use_embeddings(texts, model_params)
    
    elif model_name == "OpenAI":
        embeddings, model_info = _generate_openai_embeddings(texts, model_params)
    
    elif model_name == "Custom Embedding CSV":
        embeddings, model_info = _load_custom_embeddings(df, model_params)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Cache embeddings
    if use_cache:
        with open(cache_file, "wb") as f:
            pickle.dump({
                "embeddings": embeddings,
                "model_info": model_info
            }, f)
            print(f"Cached embeddings to: {cache_file}")
    
    return embeddings, model_info


def _generate_sbert_embeddings(
    texts: List[str], 
    params: Dict[str, Any]
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Generate embeddings using SentenceBERT models."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("SentenceBERT (sentence-transformers) is not installed")
    
    # Get model name or use default
    model_name = params.get('model_name', 'all-MiniLM-L6-v2')
    
    # Load the model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Convert to Python list
    embeddings_list = embeddings.tolist()
    
    # Get model info
    model_info = {
        "model_name": model_name,
        "model_type": "SentenceBERT",
        "dimensions": len(embeddings_list[0]) if embeddings_list else 0,
        "model_details": str(model)
    }
    
    return embeddings_list, model_info


def _generate_use_embeddings(
    texts: List[str], 
    params: Dict[str, Any]
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Generate embeddings using Universal Sentence Encoder."""
    if not TENSORFLOW_HUB_AVAILABLE:
        raise ImportError("TensorFlow Hub is not installed")
    
    # Get model URL or use default
    model_url = params.get(
        'model_url', 
        'https://tfhub.dev/google/universal-sentence-encoder/4'
    )
    
    # Load the model
    model = hub.load(model_url)
    
    # Generate embeddings
    embeddings = model(texts).numpy()
    
    # Convert to Python list
    embeddings_list = embeddings.tolist()
    
    # Get model info
    model_info = {
        "model_name": "Universal Sentence Encoder",
        "model_type": "USE",
        "dimensions": len(embeddings_list[0]) if embeddings_list else 0,
        "model_url": model_url
    }
    
    return embeddings_list, model_info


def _generate_openai_embeddings(
    texts: List[str], 
    params: Dict[str, Any]
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Generate embeddings using OpenAI's embedding models."""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI Python package is not installed")
    
    # Get API key
    api_key = params.get('api_key')
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    # Set API key
    openai.api_key = api_key
    
    # Get model name or use default
    model_name = params.get('model_name', 'text-embedding-3-small')
    
    # Generate embeddings in batches (OpenAI has rate limits)
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        response = openai.embeddings.create(
            model=model_name,
            input=batch_texts
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    # Get model info
    model_info = {
        "model_name": model_name,
        "model_type": "OpenAI",
        "dimensions": len(all_embeddings[0]) if all_embeddings else 0
    }
    
    return all_embeddings, model_info


def _load_custom_embeddings(
    df: pd.DataFrame, 
    params: Dict[str, Any]
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Load custom pre-computed embeddings."""
    # Get embedding columns
    embedding_file = params.get('embedding_file')
    if not embedding_file:
        raise ValueError("Embedding file path is required")
    
    # Load embedding file
    embedding_df = pd.read_csv(embedding_file)
    
    # Get ID column to join on
    id_column = params.get('id_column', 'id')
    if id_column not in df.columns or id_column not in embedding_df.columns:
        raise ValueError(f"ID column '{id_column}' not found in both dataframes")
    
    # Join dataframes
    merged_df = df.merge(embedding_df, on=id_column, how='left')
    
    # Get embedding columns (all numeric columns except ID)
    embedding_columns = [
        col for col in embedding_df.columns 
        if col != id_column and pd.api.types.is_numeric_dtype(embedding_df[col])
    ]
    
    if not embedding_columns:
        raise ValueError("No numeric embedding columns found in the embedding file")
    
    # Extract embeddings
    embeddings = merged_df[embedding_columns].values.tolist()
    
    # Get model info
    model_info = {
        "model_name": "Custom Embeddings",
        "model_type": "Custom",
        "dimensions": len(embedding_columns),
        "source_file": embedding_file
    }
    
    return embeddings, model_info 
from typing import List
import openai


def embed_texts(texts: List[str], api_key: str, model: str = "text-embedding-3-small") -> List[List[float]]:
    """Generate embeddings using OpenAI."""
    openai.api_key = api_key
    embeddings: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai.embeddings.create(model=model, input=batch)
        embeddings.extend([d.embedding for d in response.data])
    return embeddings

import re
import string
from typing import Dict, List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


def get_preprocessing_options() -> Dict[str, str]:
    """
    Get available preprocessing options with descriptions.
    
    Returns:
        Dictionary mapping option keys to descriptions
    """
    return {
        "lowercase": "Convert text to lowercase",
        "remove_punctuation": "Remove punctuation",
        "remove_numbers": "Remove numbers",
        "remove_whitespace": "Remove extra whitespace",
        "remove_stopwords": "Remove common stopwords",
        "lemmatize": "Lemmatization (convert words to base form)",
        "stem": "Stemming (reduce words to root form)",
        "remove_short_words": "Remove short words (<=2 characters)"
    }


def preprocess_text(text: str, options: Dict[str, bool], custom_stopwords: Optional[List[str]] = None) -> str:
    """
    Preprocess text based on selected options.
    
    Args:
        text: The input text to preprocess
        options: Dictionary of preprocessing options (option -> bool)
        custom_stopwords: Optional list of custom stopwords to remove
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    processed_text = text
    
    # Apply selected preprocessing steps
    if options.get("lowercase", False):
        processed_text = processed_text.lower()
    
    if options.get("remove_punctuation", False):
        processed_text = processed_text.translate(str.maketrans("", "", string.punctuation))
    
    if options.get("remove_numbers", False):
        processed_text = re.sub(r'\d+', '', processed_text)
    
    if options.get("remove_whitespace", False):
        processed_text = " ".join(processed_text.split())
    
    # Tokenize for word-level operations
    tokens = word_tokenize(processed_text)
    
    if options.get("remove_stopwords", False):
        stop_words = set(stopwords.words('english'))
        
        # Add custom stopwords if provided
        if custom_stopwords:
            custom_stopwords = [word.strip().lower() for word in custom_stopwords if word.strip()]
            stop_words.update(custom_stopwords)
        
        tokens = [word for word in tokens if word.lower() not in stop_words]
    
    if options.get("remove_short_words", False):
        tokens = [word for word in tokens if len(word) > 2]
    
    if options.get("lemmatize", False):
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    if options.get("stem", False):
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    
    # Join tokens back into text
    processed_text = " ".join(tokens)
    
    return processed_text


def apply_advanced_preprocessing(text: str, domain: str = "general") -> str:
    """
    Apply domain-specific advanced preprocessing.
    
    Args:
        text: The input text to preprocess
        domain: Domain type (general, scientific, legal, technical)
        
    Returns:
        Preprocessed text
    """
    # Start with basic preprocessing
    options = {
        "lowercase": True,
        "remove_punctuation": True,
        "remove_whitespace": True,
        "remove_stopwords": True
    }
    
    processed_text = preprocess_text(text, options)
    
    # Add domain-specific processing
    if domain == "scientific":
        # For scientific text, preserve numbers and certain punctuation
        options["remove_punctuation"] = False
        options["remove_numbers"] = False
        
        # Remove common scientific stopwords
        scientific_stopwords = [
            "study", "research", "data", "analysis", "method", "result",
            "significant", "sample", "figure", "table", "et", "al"
        ]
        return preprocess_text(text, options, scientific_stopwords)
    
    elif domain == "legal":
        # For legal text, preserve case and certain punctuation
        options["lowercase"] = False
        
        # Remove common legal stopwords
        legal_stopwords = [
            "section", "act", "law", "court", "case", "plaintiff", "defendant",
            "pursuant", "herein", "thereof", "whereof", "according"
        ]
        return preprocess_text(text, options, legal_stopwords)
    
    elif domain == "technical":
        # For technical text, preserve numbers and certain symbols
        options["remove_numbers"] = False
        
        # Remove common technical stopwords
        technical_stopwords = [
            "using", "use", "used", "can", "may", "also", "one", "two",
            "system", "function", "model", "value", "example"
        ]
        return preprocess_text(text, options, technical_stopwords)
    
    return processed_text 
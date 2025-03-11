import json
import string
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable

from modules.ai_assistant.feedback import FeedbackSystem
from modules.ai_assistant.analyzers import analyze_data_quality, analyze_embeddings, analyze_clusters
from modules.ai_assistant.recommendations import (
    recommend_preprocessing, 
    recommend_reference_labels,
    recommend_weighting,
    recommend_clustering
)


class AssistantAgent:
    """
    AI Assistant agent that provides guidance and recommendations throughout the application.
    """
    
    def __init__(self):
        """Initialize the assistant agent."""
        self.history = []
        self.feedback_system = FeedbackSystem()
        self.current_state = {}
    
    def get_page_guidance(self, page: str, mode: str, session_state: Dict) -> Dict[str, Any]:
        """
        Get guidance and recommendations for the current page.
        
        Args:
            page: Current application page
            mode: Assistant mode (Basic, Detailed, Expert, Off)
            session_state: Current Streamlit session state
        
        Returns:
            Dictionary with guidance message and actions
        """
        # Update current state
        self.current_state = session_state
        
        # Record page visit
        self._record_page_visit(page)
        
        # Get page-specific guidance
        if page == "Data Upload":
            return self._get_data_upload_guidance(mode)
        elif page == "Preprocessing":
            return self._get_preprocessing_guidance(mode)
        elif page == "Embedding Generation":
            return self._get_embedding_guidance(mode)
        elif page == "Dimension Analysis":
            return self._get_dimension_analysis_guidance(mode)
        elif page == "Dimension Weighting":
            return self._get_weighting_guidance(mode)
        elif page == "Clustering":
            return self._get_clustering_guidance(mode)
        elif page == "Visualization & Analysis":
            return self._get_visualization_guidance(mode)
        else:
            return {
                "message": "Select a step from the navigation to begin your analysis journey.",
                "actions": []
            }
    
    def _record_page_visit(self, page: str) -> None:
        """Record a page visit in the history."""
        self.history.append({
            "page": page,
            "timestamp": pd.Timestamp.now()
        })
    
    def _get_data_upload_guidance(self, mode: str) -> Dict[str, Any]:
        """Get guidance for the Data Upload page."""
        if 'data' not in self.current_state or self.current_state['data'] is None:
            return {
                "message": "Upload a CSV or Excel file containing your data. " +
                           "The file should have a column with labels and a column with text descriptions.",
                "actions": []
            }
        
        # Data has been uploaded
        df = self.current_state['data'].get('df')
        label_col = self.current_state['data'].get('label_col')
        desc_col = self.current_state['data'].get('desc_col')
        
        if not all([df is not None, label_col, desc_col]):
            return {
                "message": "Select the columns containing labels and descriptions, then click 'Validate and Continue'.",
                "actions": []
            }
        
        # Analyze data quality
        analysis = analyze_data_quality(df, label_col, desc_col)
        
        if mode == "Basic":
            return {
                "message": f"Your data has been loaded successfully! " +
                           f"You have {len(df)} rows and {df[label_col].nunique()} unique labels. " +
                           f"Proceed to the Preprocessing step to prepare your text data.",
                "actions": []
            }
        else:  # Detailed or Expert
            return {
                "message": analysis["message"],
                "actions": analysis["suggestions"]
            }
    
    def _get_preprocessing_guidance(self, mode: str) -> Dict[str, Any]:
        """Get guidance for the Preprocessing page."""
        if 'data' not in self.current_state or self.current_state['data'] is None:
            return {
                "message": "Please upload and validate your data first.",
                "actions": []
            }
        
        if 'df_processed' in self.current_state['data']:
            return {
                "message": "Your text has been preprocessed. You can proceed to the Embedding Generation step.",
                "actions": []
            }
        
        df = self.current_state['data'].get('df')
        desc_col = self.current_state['data'].get('desc_col')
        
        if mode == "Basic":
            return {
                "message": "Select the preprocessing options you'd like to apply to your text data. " +
                           "Common choices include lowercase conversion, removing punctuation, and stopword removal.",
                "actions": []
            }
        else:  # Detailed or Expert
            recommendations = recommend_preprocessing(df, desc_col)
            
            action_id = "apply_recommended_preprocessing"
            
            return {
                "message": recommendations["message"],
                "actions": [{
                    "id": action_id,
                    "label": "Apply Recommended Settings",
                    "callback": lambda: self._set_recommended_preprocessing(recommendations["recommended_options"])
                }]
            }
    
    def _get_embedding_guidance(self, mode: str) -> Dict[str, Any]:
        """Get guidance for the Embedding Generation page."""
        if 'data' not in self.current_state or 'df_processed' not in self.current_state['data']:
            return {
                "message": "Please complete the preprocessing step first.",
                "actions": []
            }
        
        if 'embeddings' in self.current_state and self.current_state['embeddings'] is not None:
            model_info = self.current_state['embeddings'].get('model_info', {})
            model_name = model_info.get('model_name', 'the selected model')
            dimensions = model_info.get('dimensions', 'multiple')
            
            return {
                "message": f"Embeddings generated successfully using {model_name} " +
                           f"with {dimensions} dimensions. You can now proceed to Dimension Analysis.",
                "actions": []
            }
        
        if mode == "Basic":
            return {
                "message": "Select an embedding model to convert your text descriptions into vectors. " +
                           "SentenceBERT is a good default choice for most text data.",
                "actions": []
            }
        else:  # Detailed or Expert
            return {
                "message": "Choose an embedding model based on your needs:\n\n" +
                           "• SentenceBERT: Good general-purpose embeddings, balanced speed/quality\n" +
                           "• Universal Sentence Encoder: Excellent for semantic similarity tasks\n" +
                           "• OpenAI: High quality but requires an API key and costs money per request\n\n" +
                           "For larger datasets, consider models with caching enabled.",
                "actions": []
            }
    
    def _get_dimension_analysis_guidance(self, mode: str) -> Dict[str, Any]:
        """Get guidance for the Dimension Analysis page."""
        if 'embeddings' not in self.current_state or self.current_state['embeddings'] is None:
            return {
                "message": "Please generate embeddings first.",
                "actions": []
            }
        
        if 'dimension_importance' in self.current_state and self.current_state['dimension_importance'] is not None:
            return {
                "message": "Dimension analysis complete! You've identified the most important dimensions " +
                           "for distinguishing between your reference labels. Proceed to Dimension Weighting " +
                           "to emphasize these key dimensions.",
                "actions": []
            }
        
        df = self.current_state['data'].get('df')
        label_col = self.current_state['data'].get('label_col')
        embeddings = self.current_state['embeddings'].get('vectors')
        
        if all([df is not None, label_col, embeddings]):
            recommended_labels = recommend_reference_labels(df, embeddings, label_col)
            
            if not recommended_labels:
                return {
                    "message": "Select two reference labels that represent distinct categories you want to separate.",
                    "actions": []
                }
            
            label1, label2, distance = recommended_labels[0]
            
            if mode == "Basic":
                return {
                    "message": f"I recommend using '{label1}' and '{label2}' as reference labels " +
                               f"as they appear to be well-separated in the embedding space.",
                    "actions": []
                }
            else:  # Detailed or Expert
                return {
                    "message": f"For best results, select '{label1}' and '{label2}' as reference labels. " +
                               f"These labels have a separation distance of {distance:.2f} in the embedding space, " +
                               f"making them ideal for identifying important dimensions. " +
                               f"Using distinct labels helps highlight the dimensions that best distinguish your data.",
                    "actions": []
                }
        
        return {
            "message": "Select two reference labels to analyze which dimensions are most important for distinguishing between them.",
            "actions": []
        }
    
    def _get_weighting_guidance(self, mode: str) -> Dict[str, Any]:
        """Get guidance for the Dimension Weighting page."""
        if 'dimension_importance' not in self.current_state or self.current_state['dimension_importance'] is None:
            return {
                "message": "Please complete dimension analysis first.",
                "actions": []
            }
        
        if 'weighted_embeddings' in self.current_state and self.current_state['weighted_embeddings'] is not None:
            return {
                "message": "Weighting applied successfully! Proceed to Clustering to discover patterns in your data.",
                "actions": []
            }
        
        dimension_importance = self.current_state['dimension_importance']
        
        recommendations = recommend_weighting(dimension_importance)
        
        if mode == "Basic":
            return {
                "message": f"I recommend using the {recommendations['scheme']} weighting scheme " +
                           f"with the top {recommendations['top_n']} dimensions.",
                "actions": []
            }
        else:  # Detailed or Expert
            return {
                "message": recommendations["message"],
                "actions": [{
                    "id": "apply_recommended_weighting",
                    "label": "Apply Recommended Weighting",
                    "callback": lambda: self._set_recommended_weighting(recommendations)
                }]
            }
    
    def _get_clustering_guidance(self, mode: str) -> Dict[str, Any]:
        """Get guidance for the Clustering page."""
        if 'weighted_embeddings' not in self.current_state or self.current_state['weighted_embeddings'] is None:
            return {
                "message": "Please complete dimension weighting first.",
                "actions": []
            }
        
        if 'clusters' in self.current_state and self.current_state['clusters'] is not None:
            cluster_count = len(set(self.current_state['clusters'].get('labels', [])))
            return {
                "message": f"Clustering complete! You've identified {cluster_count} clusters in your data. " +
                           f"Proceed to Visualization & Analysis to explore your results.",
                "actions": []
            }
        
        weighted_embeddings = self.current_state['weighted_embeddings']
        
        recommendations = recommend_clustering(weighted_embeddings)
        
        if mode == "Basic":
            return {
                "message": f"I recommend trying {recommendations['algorithm']} clustering " +
                           f"with {recommendations['n_clusters']} clusters.",
                "actions": []
            }
        else:  # Detailed or Expert
            return {
                "message": recommendations["message"],
                "actions": [{
                    "id": "apply_recommended_clustering",
                    "label": "Apply Recommended Clustering",
                    "callback": lambda: self._set_recommended_clustering(recommendations)
                }]
            }
    
    def _get_visualization_guidance(self, mode: str) -> Dict[str, Any]:
        """Get guidance for the Visualization & Analysis page."""
        if 'data' not in self.current_state or 'df_with_clusters' not in self.current_state['data']:
            return {
                "message": "Please complete clustering first.",
                "actions": []
            }
        
        df_with_clusters = self.current_state['data'].get('df_with_clusters')
        cluster_column = 'cluster'
        unique_clusters = sorted(df_with_clusters[cluster_column].unique())
        
        if mode == "Basic":
            return {
                "message": f"Your data has been clustered into {len(unique_clusters)} groups. " +
                           f"Explore different visualization types to understand the patterns in your data.",
                "actions": []
            }
        else:  # Detailed or Expert
            # Analyze clusters
            analysis = analyze_clusters(
                df_with_clusters, 
                self.current_state['weighted_embeddings'], 
                self.current_state['clusters']['labels'],
                self.current_state['dimension_importance']
            )
            
            return {
                "message": analysis["message"],
                "actions": []
            }
    
    def _set_recommended_preprocessing(self, recommended_options: Dict[str, bool]) -> None:
        """Set the recommended preprocessing options in the session state."""
        # This would be called by the action button to apply recommended settings
        pass
    
    def _set_recommended_weighting(self, recommendations: Dict[str, Any]) -> None:
        """Set the recommended weighting options in the session state."""
        # This would be called by the action button to apply recommended settings
        pass
    
    def _set_recommended_clustering(self, recommendations: Dict[str, Any]) -> None:
        """Set the recommended clustering options in the session state."""
        # This would be called by the action button to apply recommended settings
        pass

    def _is_technical_text(self, text_samples: List[str]) -> bool:
        """Determine if text appears to be technical in nature."""
        # Check for indicators of technical text
        technical_indicators = [
            "algorithm", "data", "function", "method", "system", 
            "analysis", "protocol", "parameter", "value", "code"
        ]
        
        technical_score = 0
        for text in text_samples:
            text_lower = text.lower()
            for indicator in technical_indicators:
                if indicator in text_lower:
                    technical_score += 1
        
        # Return True if a significant portion of samples have technical indicators
        return technical_score > (len(text_samples) * 0.3) 
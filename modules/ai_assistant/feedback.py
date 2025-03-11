import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class FeedbackSystem:
    """
    System for collecting and managing user feedback on assistant recommendations.
    """
    
    def __init__(self, storage_path: str = "feedback_data.json"):
        """
        Initialize the feedback system.
        
        Args:
            storage_path: Path to store feedback data
        """
        self.storage_path = storage_path
        self.feedback_data = self._load_feedback_data()
    
    def _load_feedback_data(self) -> Dict:
        """
        Load existing feedback data or create new structure.
        
        Returns:
            Dictionary containing feedback data
        """
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "recommendations": {},
                "explanations": {},
                "parameter_suggestions": {}
            }
    
    def _save_feedback_data(self) -> None:
        """Save feedback data to disk."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.feedback_data, f)
    
    def record_feedback(self, category: str, item_id: str, rating: int, comment: Optional[str] = None) -> None:
        """
        Record user feedback for a specific item.
        
        Args:
            category: Feedback category
            item_id: Identifier for the item
            rating: Numerical rating (typically 0=negative, 1=positive)
            comment: Optional comment from user
        """
        if category not in self.feedback_data:
            self.feedback_data[category] = {}
        
        if item_id not in self.feedback_data[category]:
            self.feedback_data[category][item_id] = {
                "ratings": [],
                "comments": []
            }
        
        self.feedback_data[category][item_id]["ratings"].append(rating)
        
        if comment:
            self.feedback_data[category][item_id]["comments"].append(comment)
        
        self._save_feedback_data()
    
    def get_feedback_stats(self, category: str, item_id: str) -> Dict[str, Any]:
        """
        Get feedback statistics for a specific item.
        
        Args:
            category: Feedback category
            item_id: Identifier for the item
            
        Returns:
            Dictionary with feedback statistics
        """
        if category in self.feedback_data and item_id in self.feedback_data[category]:
            data = self.feedback_data[category][item_id]
            ratings = data["ratings"]
            
            return {
                "count": len(ratings),
                "average_rating": sum(ratings) / len(ratings) if ratings else 0,
                "comments": data["comments"]
            }
        
        return {
            "count": 0,
            "average_rating": 0,
            "comments": []
        }
    
    def get_top_rated_items(self, category: str, min_count: int = 5) -> List[Dict[str, Any]]:
        """
        Get top-rated items in a category.
        
        Args:
            category: Feedback category
            min_count: Minimum number of ratings to consider
            
        Returns:
            List of top-rated items with statistics
        """
        if category not in self.feedback_data:
            return []
        
        items = []
        
        for item_id, data in self.feedback_data[category].items():
            ratings = data["ratings"]
            
            if len(ratings) >= min_count:
                avg_rating = sum(ratings) / len(ratings)
                
                items.append({
                    "item_id": item_id,
                    "average_rating": avg_rating,
                    "count": len(ratings),
                    "comments": data["comments"]
                })
        
        # Sort by average rating (descending)
        items.sort(key=lambda x: x["average_rating"], reverse=True)
        
        return items 
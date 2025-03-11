import pandas as pd
import numpy as np
import random
import streamlit as st
from modules.ai_assistant import ai_assistant
import json

def generate_sample_data(n_samples=500, n_categories=5, output_file="sample_data.csv"):
    """
    Generate sample data for testing the Description-Based Clustering application.
    
    Args:
        n_samples: Number of samples to generate
        n_categories: Number of categories to create
        output_file: Path to save the output CSV file
    
    Returns:
        The generated DataFrame and saves it to a CSV file
    """
    # Create categories with different themes
    categories = {
        "Technology": [
            "computer", "software", "hardware", "algorithm", "data", "network",
            "digital", "programming", "internet", "device", "system", "application"
        ],
        "Science": [
            "experiment", "research", "laboratory", "hypothesis", "analysis",
            "scientist", "discovery", "theory", "biology", "chemistry", "physics"
        ],
        "Business": [
            "company", "market", "finance", "investment", "strategy", "management",
            "product", "customer", "revenue", "profit", "service", "marketing"
        ],
        "Arts": [
            "painting", "music", "sculpture", "design", "creative", "artist",
            "exhibition", "culture", "performance", "theater", "visual", "gallery"
        ],
        "Health": [
            "medical", "wellness", "hospital", "doctor", "therapy", "disease",
            "treatment", "health", "patient", "medicine", "diagnosis", "symptoms"
        ]
    }
    
    # Select n_categories random categories if there are more categories than requested
    if n_categories < len(categories):
        selected_categories = random.sample(list(categories.keys()), n_categories)
        categories = {k: categories[k] for k in selected_categories}
    
    # Create description templates for each category
    templates = {
        "Technology": [
            "A {adj} {noun} designed for enhancing {verb} capabilities in modern {context}.",
            "This {noun} provides {adj} solutions for {context} {verb} challenges.",
            "Advanced {noun} technology that enables {adj} {verb} in various {context} settings."
        ],
        "Science": [
            "Research on {adj} {noun} phenomena in {context} {verb} processes.",
            "A scientific {noun} that investigates {adj} properties of {context} {verb}.",
            "Novel {adj} approach to understanding {noun} dynamics in {context} {verb}."
        ],
        "Business": [
            "A {adj} {noun} strategy for maximizing {context} {verb} efficiency.",
            "Innovative {noun} solutions for {adj} {context} {verb} optimization.",
            "Strategic {adj} {noun} implementation for improved {context} {verb}."
        ],
        "Arts": [
            "Creative {adj} {noun} expressing {context} themes through {verb} techniques.",
            "An artistic {noun} that showcases {adj} {verb} in {context} settings.",
            "Expressive {adj} {noun} featuring unique {context} {verb} elements."
        ],
        "Health": [
            "Advanced {adj} {noun} for treating {context} conditions through {verb} therapy.",
            "Medical {noun} designed for {adj} {verb} of {context} symptoms.",
            "Therapeutic {adj} {noun} approach to improving {context} {verb} health."
        ]
    }
    
    # Additional words for variety
    adjectives = ["innovative", "advanced", "modern", "efficient", "effective", 
                 "powerful", "elegant", "strategic", "comprehensive", "dynamic"]
    
    verbs = ["processing", "analyzing", "optimizing", "enhancing", "transforming", 
            "managing", "developing", "integrating", "implementing", "utilizing"]
    
    contexts = ["digital", "commercial", "industrial", "scientific", "creative", 
               "professional", "academic", "technical", "practical", "theoretical"]
    
    # Generate data
    data = []
    
    for i in range(n_samples):
        # Select random category
        category = random.choice(list(categories.keys()))
        
        # Select random template for the category
        template = random.choice(templates[category])
        
        # Select random words
        noun = random.choice(categories[category])
        adj = random.choice(adjectives)
        verb = random.choice(verbs)
        context = random.choice(contexts)
        
        # Fill template
        description = template.format(
            noun=noun, 
            adj=adj, 
            verb=verb, 
            context=context
        )
        
        # Create ID
        item_id = f"{category[:3].upper()}-{i+1:04d}"
        
        # Add to data
        data.append({
            "id": item_id,
            "category": category,
            "description": description
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some noise to make it more realistic
    # Occasionally mix terms from other categories
    for i in range(n_samples // 10):  # Add noise to 10% of samples
        idx = random.randint(0, n_samples - 1)
        other_category = random.choice([c for c in categories.keys() if c != df.iloc[idx]["category"]])
        other_term = random.choice(categories[other_category])
        
        # Add the term to the end of the description
        df.at[idx, "description"] += f" Includes {other_term} capabilities as well."
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Sample data generated and saved to {output_file}")
    
    return df

if __name__ == "__main__":
    generate_sample_data(n_samples=500, n_categories=5, output_file="sample_data.csv")

# After displaying preprocessing options
assistant_preprocessing = ai_assistant.recommend_preprocessing(df, 'description')
st.info(f"💡 Assistant: {assistant_preprocessing['message']}")

# Option to apply recommended settings
if st.button("Apply Recommended Preprocessing"):
    for option, value in assistant_preprocessing["recommended_options"].items():
        selected_options[option] = value 

# After generating embeddings
dimension_insights = ai_assistant.analyze_embeddings(
    embeddings, 
    model_info, 
    df[st.session_state.data['label_col']]
)

st.subheader("💡 Embedding Insights")
st.write(dimension_insights["explanation"])

# Recommend reference labels for dimension analysis
if page == "Dimension Analysis":
    recommended_labels = ai_assistant.recommend_reference_labels(
        df, embeddings, label_col
    )
    
    st.info(f"💡 Assistant recommends using '{recommended_labels[0][0]}' and '{recommended_labels[0][1]}' as reference labels for clear separation.") 

# After clustering
cluster_insights = ai_assistant.analyze_clusters(
    df_with_clusters, 
    embeddings, 
    clusters, 
    dimension_importance
)

st.subheader("💡 Cluster Insights")
st.write(cluster_insights["explanation"])

# Show cluster quality assessment
st.progress(cluster_insights["quality_score"] / 100)
st.write(f"Cluster Quality: {cluster_insights['quality_assessment']}")

# Specific recommendations
for recommendation in cluster_insights["recommendations"]:
    st.write(f"- {recommendation}")

class FeedbackSystem:
    def __init__(self, storage_path="feedback_data.json"):
        self.storage_path = storage_path
        self.feedback_data = self._load_feedback_data()
    
    def _load_feedback_data(self):
        """Load existing feedback data or create new"""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "recommendations": {},
                "explanations": {},
                "parameter_suggestions": {}
            }
    
    def _save_feedback_data(self):
        """Save feedback data to disk"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.feedback_data, f)
    
    def record_feedback(self, category, item_id, rating, comment=None):
        """Record user feedback for a specific item"""
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
    
    def get_feedback_stats(self, category, item_id):
        """Get feedback statistics for a specific item"""
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

# Add AI Assistant section to sidebar
st.sidebar.markdown("---")
st.sidebar.title("💡 AI Assistant")
assistant_mode = st.sidebar.radio(
    "Assistant Mode:",
    ["Basic", "Detailed", "Expert", "Off"],
    index=0
)

# Initialize AI Assistant in session state
if 'ai_assistant' not in st.session_state:
    from modules.ai_assistant.agent import AssistantAgent
    st.session_state.ai_assistant = AssistantAgent()

# Only show assistant if not turned off
if assistant_mode != "Off":
    # Get assistant based on current page
    if page in st.session_state:
        assistant_message = st.session_state.ai_assistant.get_page_guidance(
            page, 
            assistant_mode,
            st.session_state
        )
        
        st.sidebar.markdown(f"**Guidance for {page}:**")
        st.sidebar.info(assistant_message["message"])
        
        # Add any action buttons
        for action in assistant_message.get("actions", []):
            if st.sidebar.button(action["label"], key=f"assistant_action_{action['id']}"):
                # Execute action
                action["callback"]()
                st.experimental_rerun()
        
        # Add feedback mechanism
        st.sidebar.markdown("---")
        st.sidebar.caption("Was this guidance helpful?")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("👍 Yes", key="feedback_yes"):
                st.session_state.ai_assistant.feedback_system.record_feedback(
                    "page_guidance", page, 1
                )
                st.sidebar.success("Thanks for your feedback!")
        with col2:
            if st.button("👎 No", key="feedback_no"):
                st.session_state.ai_assistant.feedback_system.record_feedback(
                    "page_guidance", page, 0
                )
                st.sidebar.success("Thanks for your feedback!") 
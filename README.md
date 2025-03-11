# Description-Based Clustering App

A Streamlit application for interactive clustering of textual descriptions with advanced dimension analysis and visualization.

## Features

- **Data Processing**: Upload and preprocess datasets with labels and descriptions
- **Embedding Generation**: Generate high-quality embeddings using multiple embedding models
- **Dimension Analysis**: Identify and rank important embedding dimensions based on reference labels
- **Custom Weighting**: Apply various weighting schemes to emphasize key dimensions
- **Flexible Clustering**: Multiple clustering algorithms with parameter tuning
- **Interactive Visualization**: Explore clusters with interactive plots and analysis tools

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Description-Based-Clustering-App.git
cd Description-Based-Clustering-App
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload your dataset (CSV/Excel) containing labels and descriptions
2. Apply preprocessing options to clean and prepare your text data
3. Generate embeddings using your model of choice
4. Select reference labels to analyze embedding dimensions
5. Apply dimension weighting based on importance
6. Run clustering algorithms with your chosen parameters
7. Explore and analyze the resulting clusters through visualizations

## Project Structure

```
Description-Based-Clustering-App/
├── app.py                    # Main Streamlit application
├── modules/
│   ├── data_loader.py        # Data loading and validation
│   ├── preprocessor.py       # Text preprocessing functions
│   ├── embeddings.py         # Embedding generation models
│   ├── dimension_analysis.py # Dimension importance analysis
│   ├── weighting.py          # Dimension weighting schemes
│   ├── clustering.py         # Clustering algorithms
│   └── visualization.py      # Visualization functions
├── utils/
│   ├── cache.py              # Caching utilities
│   └── helpers.py            # Helper functions
└── examples/                 # Example datasets
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
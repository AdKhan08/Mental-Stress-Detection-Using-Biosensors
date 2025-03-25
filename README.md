# Mental Stress Detection Using Biosensors

The **Mental Stress Detection Using Biosensors** project identifies and classifies stress levels by analyzing physiological signals collected from biosensors. The project leverages advanced machine learning and deep learning models such as **Convolutional Neural Networks (CNN), Graph Neural Networks (GNN),** and **Random Forest (RF)** to analyze biosensor data. It processes data from multiple sensors, including heart rate, GSR, and body temperature, to predict stress levels with high accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Data Sources](#data-sources)
- [Technologies Used](#technologies-used)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training & Evaluation](#model-training--evaluation)
- [Visualization](#visualization)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Overview

The project includes:
- **Data Collection:** Importing and processing biosensor data from CSV files.
- **Feature Engineering:** Extracting statistical and temporal features.
- **Model Development:** Training multiple models (CNN, GNN, Random Forest) to classify stress levels.
- **Visualization:** Generating interactive plots and evaluation metrics to compare model performance.
- **Real-World Application:** The solution can be used in healthcare, workplace stress monitoring, and mental well-being platforms.

---

## Features

✅ **Multimodal Data Processing:**  
   Handle multiple types of biosensor data (heart rate, GSR, body temperature, etc.).

✅ **CNN for Temporal Pattern Recognition:**  
   Analyze time-series biosensor data to capture spatial and temporal stress patterns.

✅ **GNN for Graph-Based Analysis:**  
   Model relationships between sensor readings to detect hidden stress patterns.

✅ **Random Forest for Baseline Model:**  
   Traditional machine learning model for initial benchmark performance.

✅ **Feature Extraction & Engineering:**  
   Extract relevant features using statistical and temporal analysis.

✅ **Advanced Model Comparison:**  
   Evaluate and compare CNN, GNN, and RF models with detailed metrics.

✅ **Interactive Visualizations:**  
   Generate dynamic visual insights on stress levels, model predictions, and feature importance.

---

## Data Sources

The project uses biosensor datasets with the following attributes:

- **Heart Rate (HR):** Beats per minute (BPM) to indicate cardiovascular changes.
- **Galvanic Skin Response (GSR):** Electrical conductivity of the skin to indicate emotional arousal.
- **Body Temperature:** Changes in body temperature correlated with stress.
- **Respiration Rate:** Breathing frequency under different stress conditions.
- **Stress Level:** Annotated labels indicating the stress level (low, medium, high).

### Sample Datasets:
- [WESAD Dataset (PhysioNet)](https://physionet.org/content/wesad/1.0.0/)
- [MIT-BIH Stress Dataset](https://physionet.org/content/drivedb/1.0.0/)

*Note: The dataset can be replaced with real-time data collected from IoT or wearable devices.*

---

## Technologies Used

- **Python:** Core language for data analysis and model development.
- **Pandas:** Data manipulation and preprocessing.
- **NumPy:** Numerical computations.
- **Scikit-Learn:** ML model implementation (Random Forest).
- **TensorFlow/Keras:** CNN and GNN model development.
- **NetworkX:** Graph structure creation for GNN models.
- **Matplotlib & Seaborn:** Visualization and data analysis.
- **Plotly:** Interactive visualizations.
- **SQLite:** Database for storing and managing stress data.

---

## Installation & Setup

1. **Clone the Repository:**
```bash
git clone https://github.com/yourusername/mental-stress-detection.git
cd mental-stress-detection
```

2. **Set Up Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Required Dependencies:**
```bash
pip install -r requirements.txt
```

4. **Open Jupyter Notebook or Google Colab:**
- Run `jupyter notebook` or open the `stress_detection.ipynb` file in [Google Colab](https://colab.research.google.com/).

---

## Usage

### 1. **Data Preprocessing**
- Load biosensor data from CSV files.
- Handle missing values and outliers.
- Normalize and scale sensor readings.

### 2. **Feature Engineering**
- Extract statistical features such as mean, standard deviation, skewness, and kurtosis.
- Generate time-series features for CNN input.
- Create graph structures for GNN models.

### 3. **Model Training & Evaluation**
- **CNN Model:** Trained to capture temporal patterns in biosensor data.
- **GNN Model:** Applied for graph-based analysis of sensor relationships.
- **Random Forest:** Baseline model to compare results.

### 4. **Prediction & Evaluation**
- Evaluate models using accuracy, precision, recall, and F1-score.
- Compare models using ROC-AUC and confusion matrices.

### 5. **Visualization**
- Plot classification reports, feature importance, and model performance.

---

## Project Structure

```
mental-stress-detection/
│
├── README.md                  # Project documentation
├── requirements.txt           # Required Python packages
├── stress_detection.ipynb     # Main notebook with all models
├── stress_data.csv            # Sample dataset
├── models/                    # Saved CNN, GNN, and RF models
├── scripts/                   # Preprocessing and utility scripts
├── visualizations/            # Visual assets and plots
└── data/                      # Raw and processed datasets
```

---

## Model Training & Evaluation

### ML Models Implemented:
1. **Convolutional Neural Network (CNN):**
   - CNN processes the time-series sensor data.
   - Recognizes patterns and temporal variations in biosensor signals.
   - Architecture: Multiple Conv1D layers with max pooling, followed by dense layers for classification.

2. **Graph Neural Network (GNN):**
   - Models the relationships between sensor readings using graph structures.
   - Nodes represent sensor attributes, and edges indicate correlations between them.
   - Graph-based embeddings are generated to classify stress levels.

3. **Random Forest (RF):**
   - Baseline model for initial benchmarking.
   - Handles feature-based classification with high accuracy.

---

## CNN Model Summary
```python
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

## GNN Model Summary
```python
import networkx as nx
from stellargraph import StellarGraph
from stellargraph.layer import GCN

# Create a graph from sensor data
G = nx.Graph()
G.add_nodes_from(node_features)
G.add_edges_from(correlation_matrix)

stellar_graph = StellarGraph.from_networkx(G, node_features=node_features_df)
gcn_model = GCN(layer_sizes=[64, 32], activations=["relu", "relu"], dropout=0.5)
```

---

## Model Evaluation Results
- **CNN Accuracy:** ~92-94%
- **GNN Accuracy:** ~90-92%
- **Random Forest Accuracy:** ~87-89%

### Evaluation Metrics:
- **Accuracy:** Correct predictions over total predictions.
- **Precision:** Measure of relevance of positive predictions.
- **Recall:** Measure of sensitivity to detect stress levels.
- **F1-Score:** Harmonic mean of precision and recall.

---

## Visualization

✅ **ROC Curves:**  
   Compare model performance across different stress levels.

✅ **Confusion Matrices:**  
   Visualize classification results for CNN, GNN, and RF.

✅ **Feature Importance Plot:**  
   Identify key biosensor features influencing stress predictions.

✅ **Time-Series Pattern Recognition:**  
   Visualize CNN-detected patterns across time.

---

## Future Enhancements

🔹 **Real-Time Stress Monitoring:**  
   Integrate with IoT devices or wearable sensors for real-time data collection.  

🔹 **API Deployment:**  
   Deploy the models using Flask/FastAPI to serve real-time predictions.  

🔹 **Mobile/IoT App Integration:**  
   Develop a mobile or web-based interface for stress monitoring.  

🔹 **Hyperparameter Tuning:**  
   Optimize CNN, GNN, and RF models for better performance.  

🔹 **Automated ETL Pipeline:**  
   Set up an Apache Airflow pipeline to automate data ingestion and transformation.  

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- Special thanks to **NIT Bhopal** for facilitating the research and development of this project.
- Datasets sourced from PhysioNet and related repositories.

---

*Created by [Adnaan Khan](https://github.com/AdKhan08)*  
Feel free to contribute, raise issues, or suggest improvements! 🚀

# Hand Gesture Classification Using MediaPipe & Machine Learning

A real-time hand gesture recognition system that classifies hand gestures using MediaPipe hand landmarks and machine learning models trained on the HaGRID dataset.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-latest-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-yellow.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Results](#results)
- [Technologies Used](#technologies-used)

## 🎯 Overview

This project implements a complete machine learning pipeline for hand gesture classification using landmark-based features extracted from hand images. By leveraging **MediaPipe's** hand tracking solution, the system extracts 21 hand landmarks (63 features: x, y, z coordinates) and trains multiple classifiers to recognize various hand gestures in real-time.

The project utilizes the **HaGRID (Hand Gesture Recognition Image Dataset)** to train robust models capable of recognizing gestures such as: call, peace, fist, thumbs up/down, ok, stop, and more.

## ✨ Features

- **Real-time Hand Gesture Recognition**: Detect and classify hand gestures from webcam feed
- **Multi-hand Support**: Simultaneously track and classify gestures from both hands
- **Custom Preprocessing**: Scale and translation-invariant landmark normalization
- **Multiple ML Models**: Comparison of K-Nearest Neighbors, Logistic Regression, and Random Forest
- **Comprehensive Visualization**: Hand landmark plots and confusion matrices for model evaluation
- **Video Recording**: Save real-time predictions to video files with timestamps
- **Production-ready Pipeline**: Complete sklearn pipeline with preprocessing and classification

## 📊 Dataset

The project uses pre-extracted hand landmark data from the **HaGRID (Hand Gesture Recognition Image Dataset)**. The dataset contains:

- **25,675 samples** with 21 hand landmarks (63 features per sample)
- **Multiple gesture classes**: call, dislike, fist, four, like, mute, ok, one, palm, peace, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted, etc.
- **Imbalanced distribution**: Some gestures like "fist" and "mute" have fewer samples
- **No missing values**: Clean, preprocessed landmark data ready for training

Each sample contains:
- **x1-x21**: X-coordinates of 21 hand landmarks
- **y1-y21**: Y-coordinates of 21 hand landmarks  
- **z1-z21**: Z-coordinates (depth) of 21 hand landmarks
- **label**: Gesture class label

## 📁 Project Structure

```
Hand-Gesture-Classification/
├── main.ipynb                      # Main Jupyter notebook with complete pipeline
├── README.md                       # Project documentation
├── data/
│   └── hand_landmarks_data.csv    # Pre-extracted landmark dataset
└── output_videos/                 # Saved real-time prediction videos
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam for real-time gesture recognition

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Hand-Gesture-Classification.git
cd Hand-Gesture-Classification
```

2. **Install required packages**:
```bash
pip install opencv-python numpy pandas seaborn mediapipe matplotlib scikit-learn
```

Or use the following packages list:
```
opencv-python
numpy
pandas
seaborn
mediapipe
matplotlib
scikit-learn
```

## 🚀 Usage

### Running the Notebook

1. Open `main.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially to:
   - Load and explore the dataset
   - Visualize hand landmarks
   - Train and compare multiple models
   - Build the final production pipeline
   - Start real-time gesture recognition

### Real-time Gesture Recognition

The notebook includes a real-time inference section that:
- Opens your webcam
- Detects hand landmarks using MediaPipe
- Classifies gestures using the trained model
- Displays predictions on the video feed
- Saves the output to `output_videos/` with timestamp

**Controls**:
- Press `q` to quit the real-time recognition

## 🔬 Methodology

### 1. Data Preprocessing

**Custom Landmark Normalization** (`LandmarkNormalizer`):
- **Recentering**: Translates all landmarks so the wrist (landmark 0) is at the origin
- **Scaling**: Normalizes by the distance to the middle finger tip (landmark 12)
- **Benefits**: Makes the model invariant to hand position and scale

### 2. Data Splitting

- **80/20 train-test split** with stratified sampling to handle class imbalance
- **Shuffling enabled** to prevent bias from consecutive samples

### 3. Model Training & Hyperparameter Tuning

Three classifiers were evaluated using 5-fold cross-validation and grid search:

#### K-Nearest Neighbors (KNN)
- **Hyperparameters**: n_neighbors (3, 5, 7), weights (uniform, distance), metric (euclidean, manhattan)
- **Approach**: Distance-based classification using nearest neighbors

#### Logistic Regression
- **Hyperparameters**: C (0.1, 1, 10), penalty (l2, None), max_iter (500, 1000)
- **Approach**: Linear probabilistic classifier

#### Random Forest (Winner 🏆)
- **Hyperparameters**: n_estimators (50, 100, 200), max_depth (10, 20, None), min_samples_split (2, 5)
- **Approach**: Ensemble of decision trees
- **Selected as final model** due to superior performance

### 4. Final Pipeline

The production pipeline combines:
```python
Pipeline([
    ('normalizer', LandmarkNormalizer()),
    ('rf', RandomForestClassifier(best_params))
])
```

Trained on the **complete dataset** (train + test) for maximum performance.

### 5. Real-time Inference

Uses **MediaPipe Hands** for:
- Hand detection and landmark extraction
- Supports up to 2 hands simultaneously
- Processes webcam feed at ~20 FPS
- Displays predictions with hand labels (Left/Right)

## 📈 Model Performance

The Random Forest classifier achieved the best performance across all metrics:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **~98%** | **~98%** | **~98%** | **~98%** |
| K-Nearest Neighbors | ~97% | ~97% | ~97% | ~97% |
| Logistic Regression | ~93% | ~93% | ~93% | ~93% |

*Note: Exact metrics depend on hyperparameter tuning results*

### Key Insights:
- Random Forest handles the multi-class classification problem exceptionally well
- Landmark normalization significantly improves model generalization
- The model performs well even on imbalanced classes
- Common confusions occur between similar gestures (e.g., two_up variations)

## 🎨 Results

The project includes comprehensive visualizations:

1. **Hand Landmark Plots**: Visual representation of 21 hand keypoints with connections
2. **Gesture Comparison**: Side-by-side visualization of different gesture classes
3. **Confusion Matrices**: Detailed performance analysis for each model
4. **Real-time Predictions**: Live webcam feed with gesture labels and confidence

### Sample Visualizations:
- Hand skeleton overlay on detected gestures
- Distribution plots showing class imbalance
- Normalized vs. raw landmark comparisons
- Confusion matrices highlighting model strengths/weaknesses

## 🛠 Technologies Used

- **Python 3.8+**: Core programming language
- **MediaPipe**: Hand landmark detection and tracking
- **OpenCV**: Video capture and image processing
- **scikit-learn**: Machine learning models and pipeline
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👨‍💻 Author

Created as part of the Machine Learning course at ITI.

## 🙏 Acknowledgments

- **HaGRID Dataset**: For providing comprehensive hand gesture data
- **MediaPipe**: For the robust hand tracking solution
- **scikit-learn**: For machine learning tools and utilities

---

⭐ Star this repository if you find it helpful!
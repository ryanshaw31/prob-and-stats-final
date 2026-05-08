# Steam Game "Hit" Prediction Model
Probability and Statistics Final Project

A ProbStats Final project that predicts whether a Steam game will achieve **"Overwhelmingly Positive"** status using gameplay, pricing, engagement, and review-based features.

---

## Project Overview

This project uses a **Random Forest Classifier** to analyze Steam game data and predict whether a game becomes a major success ("hit") based on factors such as:

- Peak concurrent players (CCU)
- DLC count
- Achievement count
- Genre saturation
- Platform availability
- Metacritic score (w/ proxy)
- Price tiers
- Game age

---

## Technologies Used

```bash
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
```

---

## Installation
Make sure you install the following libraries in order to properly run the project:
```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Dataset

The project uses: steam_top_games_2026.csv

```bash
https://www.kaggle.com/datasets/patelris/steam-top-1495-games-dataset
```

---

### Model Training

The project trains a:

```bash
RandomForestClassifier
```

with:

- 500 n_estimators
- Balanced subsampling
- Stratified train/test split (70% Training : 30% Testing)
- Tuned prediction threshold

---

## Model Evaluation

The model outputs:

- Accuracy Score
- Classification Report
- Precision-Recall Curves
- Confusion Matrix
- Feature Importance Rankings

---

## Data Visualizations

The project generates several visualizations including:

- CCU engagement distributions
- Precision vs Recall comparison
- Genre frequency analysis
- Price tier distribution
- Feature importance rankings
- Confusion matrix heatmap

---

## Example Output

<img width="384" height="420" alt="image" src="https://github.com/user-attachments/assets/6331fd08-5966-456f-ad78-63b9e76a5f15" />

---

## Author

Project by Ryan Shaw and Christian Torrazo

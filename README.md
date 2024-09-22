# Anomaly Detection in Networks Using Machine Learning

This project implements a machine learning pipeline to detect network anomalies (such as cyberattacks) in the **CICIDS2017 dataset**. Several machine learning algorithms are applied to classify network traffic as benign or malicious. The project focuses on feature selection, model evaluation, and performance comparison across different algorithms.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Feature Selection](#feature-selection)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Performance Metrics](#performance-metrics)
- [How to Run](#how-to-run)
- [Results](#results)
- [Citations](#citations)

## Introduction
The goal of this project is to detect anomalous network traffic (cyberattacks) using machine learning. It processes network data and applies various models to identify attacks. The CICIDS2017 dataset is used for this purpose, containing several attack types, including DDoS, DoS, brute force, and web attacks.

The project follows these key steps:
1. Data preprocessing and cleaning.
2. Feature selection to identify important attributes in network traffic.
3. Application of machine learning algorithms to classify benign and attack traffic.
4. Evaluation and comparison of models based on performance metrics.

## Dataset
The dataset used in this project is the **CICIDS2017** dataset. It contains both benign and malicious network traffic. You can download the dataset from [here](https://www.unb.ca/cic/datasets/ids-2017.html).

Ensure that all required CSV files are in the `CSVs/` folder in the root directory.

## Dependencies
The project is implemented using **Python 3.6** and relies on the following libraries:
- [Pandas](https://pandas.pydata.org/) — for data manipulation and analysis.
- [NumPy](https://numpy.org/) — for numerical operations.
- [Matplotlib](https://matplotlib.org/) — for generating graphs and visualizations.
- [Scikit-learn](https://scikit-learn.org/) — for machine learning algorithms and feature selection.

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Project Structure
The project consists of several scripts that handle different parts of the pipeline:

```
.
├── attacks/                      # Contains individual CSV files for each attack type
├── CSVs/                         # Contains original CSV files (CICIDS2017 dataset)
├── results/                      # Directory where results are stored
├── feature_pics/                 # Directory where feature importance graphs are stored
├── result_graph_3/               # Directory where F1-score boxplots are saved
├── preprocessing.py              # Script for data preprocessing and cleaning
├── feature_selection.py          # Script for feature selection using RandomForestRegressor
├── ml_implementation_attack.py   # Machine learning on individual attack files
├── ml_implementation_all.py      # Machine learning on the combined dataset (all_data.csv)
├── README.md                     # This file
└── requirements.txt              # Contains Python dependencies
```

## Feature Selection
The **feature selection** process identifies the most important features for detecting anomalies. The RandomForestRegressor is used to rank the importance of features based on their contribution to the classification of benign and attack traffic.

Feature importance graphs are generated for each attack type and saved in the `feature_pics/` folder.

## Machine Learning Algorithms
The following machine learning algorithms are applied to the dataset:
- **Naive Bayes**
- **Quadratic Discriminant Analysis (QDA)**
- **Random Forest**
- **ID3 (Decision Tree)**
- **AdaBoost**
- **Multi-Layer Perceptron (MLP)**
- **K-Nearest Neighbors (KNN)**

The models are evaluated using **10-fold cross-validation**, and performance metrics are reported for each algorithm.

## Performance Metrics
The performance of each model is evaluated based on the following metrics:
- **Accuracy**: The proportion of correct classifications.
- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly.
- **F1-score**: The harmonic mean of precision and recall.
- **Execution time**: The time taken to train and test each model.

These metrics are printed to the console and saved in CSV files (`results_3.csv` for the 7-feature model, `results_Final.csv` for the full dataset).

## How to Run
1. **Preprocess the data**:
   Run the preprocessing script to clean and combine the dataset:
   ```bash
   python preprocessing.py
   ```

2. **Perform feature selection**:
   Run the feature selection script to calculate the importance of features:
   ```bash
   python feature_selection.py
   ```

3. **Run machine learning models**:
   Apply machine learning models to detect anomalies in the network traffic:
   - For individual attack files:
     ```bash
     python ml_implementation_attack.py
     ```
   - For the full dataset:
     ```bash
     python ml_implementation_all.py
     ```

4. **Check results**:
   Results will be saved in the `results/` folder. Boxplots showing the F1-scores for each model will be saved in the `result_graph_3/` folder.

## Results
Each machine learning algorithm is evaluated based on its performance metrics. The CSV files (`results_3.csv`, `results_Final.csv`) provide detailed metrics for each algorithm across multiple runs. Additionally, boxplots show the F1-score distribution for each algorithm.


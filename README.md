# Sustainable Fuel Blend Property Predictor

A machine learning project developed for the **Shell AI Hackathon for Sustainable and Efficient Energy**, designed to predict the properties of complex fuel blends using a **CatBoost model** and an **interactive Streamlit application**.

---

## Project Overview

The global push for sustainability is transforming the energy sector. **Sustainable Aviation Fuels (SAFs)** are at the forefront of this change, offering a pathway to significantly reduce the environmental footprint of the aviation industry.

However, creating optimal fuel blends is a complex challenge. The relationships between component fractions and final blend properties are non-linear, involving synergistic effects that are hard to model using traditional methods.

This project addresses that challenge by building a robust predictive system that can accurately forecast the final properties of a fuel blend from its constituent components. This tool aims to **accelerate R&D for sustainable fuels**, reducing development time and cost.

---

## Solution

This repository contains two main components:

### 1. Predictive Machine Learning Model
- A **CatBoost regressor** trained on a dataset of complex fuel blends.
- Incorporates **extensive feature engineering** to capture nuanced relationships between components.

### 2. Interactive Web Application
- A **Streamlit-based dashboard** for uploading custom blend data.
- Provides **instant predictions** of final blend properties using the trained model.

---

## Key Features

### Advanced Feature Engineering
Over **150 features** designed to model key blend dynamics:
- **Weighted Blend Properties**: Expected output properties based on component-wise contributions.
- **Component Dominance**: Influence of primary base fuels (e.g., Component 4).
- **Chemical Synergy**: Interaction terms between specific component pairs.
- **Blend Complexity**: Metrics like entropy and Gini coefficient to measure diversity.

### Accurate Predictions
- Uses **MultiOutputRegressor with CatBoost** for high-performance, multi-target regression.

### Interactive User Interface
- Built with **Streamlit** for accessibility.
- Supports **CSV upload**, real-time **predictions**, and **result visualization**.

### Feature Importance Analysis
- Visualizes the **top 20 most influential features**.
- Offers **interpretability** into the model’s decision-making process.

---

## Technical Stack

| Category         | Tools Used               |
|------------------|--------------------------|
| Language          | Python                   |
| Data Processing   | Pandas, NumPy            |
| Machine Learning  | CatBoost, Scikit-learn   |
| Web Framework     | Streamlit                |
| Model I/O         | Joblib                   |

---


### Prerequisites
- Python 3.9 or higher
- Catboost gradient model
- Streamlit
- Git

---

### 1. Clone the Repository

```bash

Fuel-Blend-Prediction-Model/
├── .gitignore                    # Git ignored files
├── README.md                     # This file
├── app.py                        # Streamlit application
├── requirements.txt              # Python dependencies
├── train_model.py                # Model training script
├── dataset/
│   └── train.csv                 # Training dataset
└── catboost_fuel_blend_model.joblib  # (Generated after training)
```

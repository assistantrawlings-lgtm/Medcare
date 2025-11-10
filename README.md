# Medcare
I served as the Data Analyst for Medcare, a medtech project by Studio3lauchpad's Team 2, a cross-functional unit comprising Product Managers, UI/UX Designers, and Frontend/Backend Engineers.
My primary role was to define the medical triage rules and translate them into a production-ready model capable of assigning an urgency level to patients based on their symptoms.
Key Contributions:
Triage Rule Operationalization: Developed a classification pipeline to predict both the Triage Color (Green, Yellow, Red) and Triage Level (e.g., Severe - Seek Immediate Care).
Model Training & Performance: Trained and evaluated multiple classification algorithms using labeled symptom data, including Logistic Regression, Decision Tree, and Random Forest.
Optimal Solution: Implemented Logistic Regression, achieving an optimal prediction accuracy of 88.89% for assigning the immediate care urgency level (Triage_Color), ensuring reliable patient guidance.
Data Preparation: Handled end-to-end data processing, including data cleaning and applying Label Encoding to transform patient symptoms into a model-consumable feature set.
This foundational work established the core logic for Medcare's automated patient recommendation system.


## Medcare Triage Rule - Classification Pipeline
This notebook implements a classification pipeline to predict the Triage Level and Triage Color based on patient symptoms using a medical triage rule dataset.
1. Import Libraries
| Cell | Code |
|---|---|
| Code 11 | ```python |
===========================================
Medcare Triage Rule - Full Pipeline
===========================================
##--- 1. IMPORT LIBRARIES ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib # Used for saving models

## 2. Data Loading and Cleaning

### Load Data

| Cell | Code | Output |
| :--- | :--- | :--- |
| **Code 12** | ```python\n# load data\ndata = pd.read_excel('data/Medcare Triage Rule.xlsx')\ndata\n``` | The initial DataFrame includes `Symptoms`, `Triage_Color`, `Triage_Level`, and several empty/NaN columns (`Unnamed: 3` to `Unnamed: 6`). |
| **Code 13** | ```python\ndata.dropna(axis=1, inplace=True)\n``` | *Drops columns with all NaN values.* |
| **Code 14** | ```python\ndata\n``` | **Cleaned DataFrame (Snippet)**: |

| | Symptoms | Triage_Color | Triage_Level |
| :---: | :--- | :--- | :--- |
| **0** | Fever | Yellow | Moderate - Visit Clinic Soon |
| **1** | Severe Headache | Yellow | Moderate - Visit Clinic Soon |
| **2** | Low Blood Sugar | Red | Severe - Seek Immediate Care |
| **3** | High Blood Sugar | Red | Severe - Seek Immediate Care |
| **4** | Muscle Pain | Yellow | Moderate - Visit Clinic Soon |
| **...** | ... | ... | ... |
| **56** | Loss Of Speech | Yellow | Moderate - Visit Clinic Soon |
| **57** | Abdominal Pain | Yellow | Moderate - Visit Clinic Soon |
| **58** | Infertility | Yellow | Moderate - Visit Clinic Soon |

### Data Inspection

| Cell | Code | Output |
| :--- | :--- | :--- |
| **Code 15** | ```python\ndata.dtypes\n``` | ```\nSymptoms object\nTriage_Color object\nTriage_Level object\ndtype: object\n``` |
| **Code 16** | ```python\nprint("\nColumns:", data.columns.tolist())\n``` | ```\nColumns: ['Symptoms', 'Triage_Color', 'Triage_Level']\n``` |
| **Code 17** | ```python\ndata.isnull().sum()\n``` | ```\nSymptoms 0\nTriage_Color 0\nTriage_Level 0\ndtype: int64\n``` |

## 3. Feature Encoding

The categorical features (`Symptoms`, `Triage_Color`, `Triage_Level`) are encoded using `LabelEncoder`.

| Cell | Code | Output |
| :--- | :--- | :--- |
| **Code 19** | ```python\n# Encode categorical features\nLEC_Symptoms = LabelEncoder()\nLEC_Triage_Color = LabelEncoder()\nLEC_Triage_Level = LabelEncoder()\n``` | *Encoder objects initialized.* |
| **Code 22** | ```python\ndata['Symptoms'] = LEC_Symptoms.fit_transform(data['Symptoms'])\ndata\n``` | *The `Symptoms` column is converted to numerical labels.* |
| **Code 23** | ```python\ndata['Triage_Color'] = LEC_Triage_Color.fit_transform(data['Triage_Color'])\ndata\n``` | *The `Triage_Color` column is converted to numerical labels (e.g., Green=0, Red=1, Yellow=2).* |
| **Code 25** | ```python\ndata['Triage_Level'] = LEC_Triage_Level.fit_transform(data['Triage_Level'])\ndata\n``` | *The final target `Triage_Level` is converted to numerical labels (e.g., Mild=0, Moderate=1, Severe=2).* |

**Final Encoded DataFrame (Snippet)**

| | Symptoms | Triage_Color | Triage_Level |
| :---: | :--- | :--- | :--- |
| **0** | 18 | 2 | 1 |
| **1** | 45 | 2 | 1 |
| **2** | 28 | 1 | 2 |
| **3** | 22 | 1 | 2 |
| **4** | 31 | 2 | 1 |
| **...** | ... | ... | ... |
| **56** | 27 | 2 | 1 |
| **57** | 0 | 2 | 1 |
| **58** | 24 | 2 | 1 |

## 4. Model Training Setup

| Cell | Code | Output |
| :--- | :--- | :--- |
| **Code 26** | ```python\n# Define features (X) and target (y)\nX = data[['Symptoms']]\ny = data['Triage_Color'] # Triage_Color as the target\ny_level = data['Triage_Level'] # Triage_Level as an alternative target\ndata\n``` | *Features and targets defined.* |
| **Code 27** | ```python\n# Split data for Triage_Color prediction\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\nprint("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)\n``` | ```\nTrain shape: (41, 1) Test shape: (18, 1)\n``` |
| **Code 33** | ```python\n# --- MODEL TRAINING & EVALUATION ---\ndef evaluate_model(model, model_name):\n    model.fit(X_train, y_train)\n    y_pred = model.predict(X_test)\n    \n    if set(y) == {0, 1, 2}: # Classification metrics\n        acc = metrics.accuracy_score(y_test, y_pred)\n        print(f"{model_name} Accuracy: {acc:.4f}")\n        print(metrics.classification_report(y_test, y_pred))\n    else: # Regression metrics\n        mse = metrics.mean_squared_error(y_test, y_pred)\n        print(f"{model_name} MSE: {mse:.4f}")\n``` | *Model evaluation function defined.* |

## 5. Model Evaluation (Target: `Triage_Color`)

| Cell | Code | Output |
| :--- | :--- | :--- |
| **Code 34** | ```python\n# Evaluate Logistic Regression\nLogisticRegression = LogisticRegression(random_state=42)\nevaluate_model(LogisticRegression, 'Logistic Regression')\n``` | ```\nLogistic Regression Accuracy: 0.8889\n              precision    recall  f1-score   support\n\n           0       1.00      1.00      1.00         3\n           1       0.83      0.83      0.83         6\n           2       0.91      0.91      0.91         9\n\n    accuracy                           0.8889        18\n   macro avg       0.91      0.91      0.91        18\nweighted avg       0.8889    0.8889    0.8889        18\n``` |
| **Code 35** | ```python\n# Evaluate Decision Tree\nDecisionTreeClassifier = DecisionTreeClassifier(random_state=42)\nevaluate_model(DecisionTreeClassifier, 'Decision Tree')\n``` | ```\nDecision Tree Accuracy: 0.7222\n...\n``` |
| **Code 36** | ```python\n# Evaluate K-Neighbors Classifier\nKNeighborsClassifier = KNeighborsClassifier()\nevaluate_model(KNeighborsClassifier, 'K-Neighbors')\n``` | ```\nK-Neighbors Accuracy: 0.8889\n...\n``` |
| **Code 37** | ```python\n# Evaluate Naive Bayes\nGaussianNB = GaussianNB()\nevaluate_model(GaussianNB, 'Naive Bayes')\n``` | ```\nNaive Bayes Accuracy: 0.8333\n...\n``` |
| **Code 38** | ```python\n# Evaluate Support Vector Machine\nSVC = SVC(random_state=42)\nevaluate_model(SVC, 'Support Vector Machine')\n``` | ```\nSupport Vector Machine Accuracy: 0.8889\n...\n``` |
| **Code 39** | ```python\n# Evaluate Random Forest\nRandomForestClassifier = RandomForestClassifier(random_state=42)\nevaluate_model(RandomForestClassifier, 'Random Forest')\n``` | ```\nRandom Forest Accuracy: 0.8889\n...\n``` |
| **Code 40** | ```python\nprint("✅ All models evaluated.")\n``` | ```\n✅ All models evaluated.\n``` |

## 6. Model Saving

| Cell | Code | Output |
| :--- | :--- | :--- |
| **Code 45** | ```python\n# Save model and encoders\njoblib.dump(LogisticRegression, "best_model.pkl")\njoblib.dump(LEC_Symptoms, "LEC_Symptoms.pkl")\njoblib.dump(LEC_Triage_Color, "LEC_Triage_Color.pkl")\njoblib.dump(LEC_Triage_Level, "LEC_Triage_Level.pkl")\n\nprint("✅ Model and encoders saved successfully!")\n``` | ```\n✅ Model and encoders saved successfully!\n``` |


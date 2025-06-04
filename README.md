# 🛳 Titanic Survival Prediction (Kaggle)

This project is a solution to the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) challenge on Kaggle. It predicts passenger survival based on features like age, sex, passenger class, and more, using a Random Forest classifier.

---

## 🚀 Features

- Data cleaning and preprocessing
- Feature engineering:
  - Extracting titles from names
  - Creating family size
- Label encoding of categorical features
- Model training using RandomForestClassifier
- Cross-validation for performance evaluation
- Generates submission CSV for Kaggle

---

## 📂 Project Structure

titanic-survival/ ├── train.csv # Training dataset ├── test.csv # Test dataset ├── titanic_submission.csv # Submission file with predictions ├── titanic.py # Python script for training the model

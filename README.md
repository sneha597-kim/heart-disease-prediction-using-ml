# 🫀 Heart Disease Prediction Web Application

## 📌 Overview
This web application predicts the likelihood of heart disease based on patient health parameters using a trained **Support Vector Machine (SVM)** model.  
It is designed for doctors to input patient details, get instant predictions, and store patient records for future reference.

---

## 🚀 Features
- **Doctor Authentication** – Login & Registration via mobile number or email using Flask-Login.
- **Home Dashboard** – Simple interface with a "Get Started" button.
- **Patient Data Input** – Form to collect health metrics such as age, gender, cholesterol, resting BP, etc.
- **ML Model Prediction** – Uses a trained SVM model (`refined_best_svm_model.pkl`) to predict heart disease risk.
- **Recommendation System** – Advises consultation with a specialist if severe risk is detected.
- **Patient Data Storage** – Saves patient records in an SQLite database for later review.
- **Profile Page** – Displays logged-in doctor’s details.

---

## 🛠️ Tech Stack
**Frontend**: HTML, CSS, JavaScript  
**Backend**: Python (Flask), SQLite  
**Machine Learning**: SVM with preprocessing using StandardScaler  
**Authentication**: Flask-Login  

---

---

## 📊 Machine Learning Model
- **Algorithm**: Support Vector Machine (SVM)
- **Best Kernel**: RBF
- **Preprocessing**: StandardScaler for feature normalization
- **Training Data**: Cardiovascular dataset with features like:
  - Age
  - Gender
  - Chest Pain Type
  - Resting BP
  - Serum Cholesterol
  - Fasting Blood Sugar
  - Resting ECG
  - Max Heart Rate
  - Exercise-Induced Angina
  - Oldpeak (ST depression)
  - Slope of ST segment
  - Number of major vessels
  - Target (0: No Heart Disease, 1: Heart Disease)
- **Performance**:
  - Accuracy: *97.5*

---



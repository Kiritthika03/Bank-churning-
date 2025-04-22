# Bank-churning-
# 🏦 Bank Customer Churn Prediction

## 📌 Project Overview
This project aims to predict whether a customer will leave the bank (churn) based on their demographic and account-related features. Using machine learning techniques, particularly an Artificial Neural Network (ANN), we build a predictive model to help banks proactively identify at-risk customers and improve retention strategies.

---

## 📂 Dataset
The dataset used is sourced from [Kaggle](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction/data) and includes 10,000 customer records with features such as:

- **Demographics**: Geography, Gender, Age
- **Account Info**: Credit Score, Balance, Estimated Salary, Tenure
- **Behavioral Indicators**: Number of Products, IsActiveMember, HasCrCard
- **Target**: `Exited` (1 = Churned, 0 = Stayed)

---

## 🛠️ Project Steps

1. **Exploratory Data Analysis (EDA)**
   - Visualized churn trends by gender, geography, age, and tenure
   - Examined feature correlations with the target variable

2. **Data Preprocessing**
   - Handled missing values
   - Normalized key numerical features
   - Applied one-hot and label encoding

3. **PCA Visualization**
   - Used PCA to assess linearity of the dataset (found to be non-linear)

4. **Model Building**
   - Built an ANN using PyTorch
   - Trained the model with `nn.BCELoss()` and the Adam optimizer
   - Used 100 epochs and batched DataLoader for efficient training

5. **Evaluation**
   - Achieved **84% accuracy**
   - Evaluated model using accuracy score, confusion matrix, and ROC-AUC

---

## 📈 Key Insight

- Customers in **Germany** showed the highest churn rates
- **Inactive members** and those with **1 or 4 products** were more likely to churn
- **ANN outperformed** traditional linear classifiers due to the non-linear nature of the dataset

---

## 💡 Future Improvements

- Implement dropout or early stopping to reduce overfitting
- Try XGBoost or ensemble methods for comparison
- Deploy model as a simple web app for real-time churn prediction

---

## 🤝 Acknowledgements

- Dataset by [Shubham Meshram](https://www.kaggle.com/datasets/shubhammeshram579)
- Tools used: Pandas, Seaborn, Matplotlib, Scikit-learn, PyTorch

---

## 📬 Contact

For questions or collaboration, feel free to reach out!


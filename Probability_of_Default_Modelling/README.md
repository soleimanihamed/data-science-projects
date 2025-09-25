# Probability of Default (PD) Modelling  

## 📌 Overview  
This project presents the end-to-end development of a machine learning-powered **Probability of Default (PD)** model. The work combines rigorous **data science methodology** with **strategic business insights**, addressing a critical question for modern lenders:  

> *How can we more accurately and efficiently predict borrower defaults while balancing risk management, operational costs, and customer experience?*  

The outcome is a high-performing **XGBoost model**, validated across multiple statistical and business metrics, offering a deployable solution for credit risk management.  

---

## 🔄 Project Workflow  

### 1. Business Understanding & Data Collection  
- Defined objectives: regulatory compliance, automated decisioning, risk-based pricing, portfolio monitoring.  
- Data sources: borrower and loan records, credit bureau scores, macroeconomic indicators.  

### 2. Data Preparation & Exploration  
- Cleaning, reconciliation, and profiling of raw data.  
- Handling missing values (>30% threshold), outlier treatment (IQR, Z-scores, Isolation Forest).  
- Exploratory Data Analysis (EDA): trend discovery, anomaly detection, correlation checks.  

### 3. Feature Engineering & Selection  
- Transformation via **supervised binning** and **Weight of Evidence (WoE)**.  
- Information Value (IV) analysis confirmed strong predictors such as:  
  - Loan Percent Income (IV = 0.95)  
  - Interest Rate (IV = 0.77)  
- Feature importance consensus across Logistic Regression, Decision Tree, Random Forest, and XGBoost identified affordability metrics as the **dominant drivers of default risk**.  

### 4. Model Development  
- Algorithms trained: Logistic Regression, Decision Tree, Random Forest, XGBoost.  
- Cross-validation and preprocessing pipeline ensured robustness, reproducibility, and regulatory traceability.  

### 5. Model Evaluation  

Comprehensive metrics were applied:  
- **AUC-ROC**, Precision, Recall, F1-Score, Log Loss.  
- Validation on unseen test data to ensure generalisation.  

| Metric        | Logistic Regression | Decision Tree | Random Forest | **XGBoost (Winner)** |
|---------------|---------------------|---------------|---------------|-----------------------|
| AUC           | 0.86                | 0.90          | 0.94          | **0.95**              |
| Accuracy      | 0.88                | 0.91          | 0.93          | **0.94**              |
| Recall        | **0.91**            | 0.81          | 0.81          | 0.85                  |
| Precision     | 0.45                | 0.55          | 0.68          | **0.75**              |
| Log Loss      | 0.39                | 0.31          | 0.24          | **0.21**              |  

**Champion:** XGBoost – superior discriminatory power, precision, and reliability.  
**Runner-up:** Random Forest – competitive but slightly weaker than XGBoost.  
**Regulatory benchmark:** Logistic Regression – indispensable for transparency.  

### 6. Business Insights  
Deployment of the XGBoost model enables:  
- **Reduced Credit Losses** – accurate identification of high-risk borrowers at origination.  
- **Lower Operational Costs** – fewer false positives (reduced manual review workload).  
- **Improved Customer Experience** – fewer incorrect rejections of creditworthy applicants.  
- **Risk-Based Pricing** – reliable probability scores for personalised loan pricing.  

---

## ✅ Key Takeaways  
- A structured, transparent, and reproducible framework was followed from **raw data → business-aligned model**.  
- Borrower **affordability metrics** (Loan Percent Income, Interest Rate, Income) emerged as the most critical predictors.  
- The final **XGBoost model** offers both **statistical excellence** and **strategic business impact**, showcasing the power of machine learning in credit risk management.  

---

## 📂 Repository Contents  
- `Input Data/` – Sample/preprocessed datasets  
- `PD Modelling Report.docx` – Project reports  

---

## 🛠️ Technologies Used  
- **Python** (Pandas, NumPy, Scikit-learn, XGBoost)  
- **Data Visualisation** (Matplotlib, Seaborn)  
- **Statistical Analysis** (EDA, IV, WoE, PCA, MCA, FAMD)  

---

## 📖 Conclusion  
This project demonstrates how advanced machine learning techniques, when embedded within a rigorous methodological framework and aligned with business goals, can transform credit risk management.  

The **Probability of Default (PD) Model** developed here stands as a deployable solution capable of improving financial resilience, regulatory compliance, and customer trust.  

---

✨ *Author: Hamed Soleimani*  

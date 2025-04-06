# AI-Powered Probability of Default (PD) Modelling: A Case Study in Credit Risk Analytics  
### Strategic Data Collection for Credit Risk Assessment

## Overview  
This project explores the foundational step in developing a robust **Probability of Default (PD)** model: **strategic data collection**. Drawing on financial domain knowledge and established risk modelling practices, it demonstrates how to gather and justify key internal, external, and derived features that are critical for accurate credit risk prediction.

The analysis is conducted using a synthetic dataset sourced from DataCamp’s *Credit Risk Modeling in Python* course. While fictional, the data simulates real-world lending portfolios and is scalable to include more advanced features in a production setting.

## Main Report  
The main report outlines the data collection strategy, dataset overview, justification of selected features, and initial exploratory steps taken before detailed data preprocessing and modelling. It sets the stage for subsequent phases of PD modelling.

## Problem Statement  
This phase of the study aims to:

- **Establish a strategic data collection framework** for PD modelling  
- **Identify and justify critical features** from internal records, macroeconomic indicators, and behavioural metrics  
- **Lay the groundwork for effective model training** by understanding the data’s structure and potential predictive power  
- **Demonstrate a real-world credit scoring pipeline** with documented rationale, aiding reproducibility and transparency

Further stages will involve **exploratory data analysis (EDA), preprocessing, model training, and evaluation**, as the project progresses toward a full credit risk solution.

## Why Does PD Matter So Much?  
- Under frameworks like **Basel III** and **IFRS 9**, PD is essential for calculating **capital adequacy** and **expected credit losses (ECL)**  
- It’s also a key driver in determining **loan pricing**, **stress testing**, and **Risk-Adjusted Return on Capital (RAROC)**

## About the Case Study  
In this study, I aimed to model PD from both an **analytical and business perspective**, ensuring the development process is purpose-driven and aligned with real-world use cases. I took on a **tri-fold role**:

- **Business Analyst** – Aligning model development with strategic goals, whether for regulatory compliance (Basel III/IFRS 9), credit scoring improvements, or risk-based pricing strategies  
- **Data Analyst** – Bridging the gap between raw data and business needs, interpreting patterns, identifying anomalies, and transforming data into actionable features that drive clarity in decision-making  
- **Data Scientist** – Designing and implementing a modular, open-source solution—**Daanish**—to operationalize the full PD modelling pipeline. This approach ensures scalability, repeatability, and interpretability, providing insights stakeholders can trust

By combining **statistical rigor** with **business context**, this case study highlights the dual importance of **model accuracy** and **real-world relevance**.

## Project Roadmap  
This repository follows a modular, step-by-step structure that reflects the real-world PD modelling pipeline:

1. **Strategic Data Collection**  
   Identify and gather internal, external, and derived features needed for PD modelling  
2. **Preliminary EDA (Exploratory Data Analysis)**  
   Understand data structure, uncover trends, and assess variable importance  
3. **Data Preprocessing**  
   Handle missing values, encode categorical features, and transform data for modelling  
4. **Feature Engineering**  
   Create business-relevant features such as loan-to-income ratios, behavior indicators, etc.  
5. **Modelling**  
   Build baseline and advanced models (Logistic Regression, Random Forest, XGBoost, etc.)  
6. **Model Evaluation**  
   Use classification metrics (AUC-ROC, precision, recall, F1) and calibration analysis  
7. **Interpretability**  
   Use SHAP values, feature importance charts, and scenario analysis for transparency  
8. **Deployment-Ready Packaging**  
   Prepare the pipeline for API integration or dashboard embedding  





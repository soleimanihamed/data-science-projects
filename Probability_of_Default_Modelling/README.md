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

1. **Data Collection & Business Understanding**  
   → Aligning data with business objectives such as compliance, pricing, and risk mitigation.

2. **Preliminary EDA**  
   → [Current Phase] Uncovering early patterns, relationships, and anomalies using histograms, box plots, scatter plots, probability distributions, and crosstabs.

3. **Data Preprocessing**  
   → Handling missing values, treating outliers, and preparing clean, model-ready data.

4. **Full EDA Post-Cleaning**  
   → In-depth exploration: correlation analysis, PCA, multicollinearity, and more.

5. **Feature Engineering & Selection**  
   → Building advanced features (e.g. early payment rate, past default indicators) and selecting the most informative ones.

6. **Model Building**  
   → Training and optimizing models: Logistic Regression, Random Forest, XGBoost, etc.

7. **Model Evaluation**  
   → Validating performance using metrics such as AUC-ROC, Precision, Recall, F1, and KS-statistic.

8. **Model Validation & Testing**  
   → Ensuring generalisability through time-split testing and stability analysis.

9. **Model Deployment & Monitoring**  
   → Preparing the pipeline for real-time or batch integration and tracking performance over time.

## Step 1: Data Collection & Business Understanding
Every robust PD modelling project begins with a clear business objective. In this study, we explore key use cases—regulatory compliance (e.g., Basel III, IFRS 9), credit decisioning, risk-based pricing, and portfolio monitoring—to shape our modelling approach. Each use case influences data requirements, model complexity, and stakeholder expectations.

We use a synthetic dataset (sourced from DataCamp) that includes borrower-level attributes such as age, income, loan intent, home ownership, debt-to-income ratio, prior default status, and loan performance. While fictional, the data captures many real-world risk indicators.

Key features are chosen based on both domain knowledge and empirical research, including:
- **Loan Intent**: Insight into borrower motivation and risk profile.
- **Loan Grade** Lender’s composite assessment of credit risk.
- **Debt-to-Income Ratio** : Indicator of financial pressure.
- **Defaulted Before** : Historical behaviour as a risk signal.
- **Interest Rate & Loan Amount** : Proxies for exposure and pricing of risk.
At this stage, no transformations or filtering are applied—data is preserved in its rawest form to retain all potentially informative signals. Preprocessing is deferred to later phases for more thoughtful, context-aware handling.

## Step 2: Preliminary EDA Highlights

In the current phase, we conducted a thorough exploratory analysis to understand relationships between key features and loan default risk. Key findings include:

- **Home Ownership**: Renters show the highest default rate (73%), while mortgage holders are more stable (24% default).
- **Loan Intent**: Medical-related loans exhibit higher default (23%), whereas venture loans show lower risk (12%).
- **Past Defaults**: Applicants with a history of default are over twice as likely to default again.
- **Combined Analysis**: Renters with prior defaults have the highest risk profile, while mortgage holders with clean histories are the most reliable.
- **Visualization Techniques Used**: Histograms, box plots, scatter plots, probability distributions, and crosstabs.

These insights guide our data preprocessing strategy and inform which features require transformation, imputation, or special handling.




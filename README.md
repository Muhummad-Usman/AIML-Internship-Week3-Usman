# 🏠 AI/ML Internship — Week 3: Data Visualization & Feature Engineering

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-11557c?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-4C72B0?style=for-the-badge)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

**Name:** Usman Asif &nbsp;|&nbsp; **Program:** Digitech Offerings — AI/ML Internship &nbsp;|&nbsp; **Week:** 3 of 8  
**Date:** 10th May, 2026 &nbsp;|&nbsp; **Instructor:** Zain Ul Abideen

</div>

---

## 📌 Project Overview

This notebook covers **Week 3** of the AI/ML Internship Program and focuses on advanced **data visualization**, **feature engineering**, **categorical encoding**, **feature scaling**, **skewness treatment**, and **feature selection** using the **House Prices — Advanced Regression Techniques** dataset from Kaggle.

The project demonstrates a complete preprocessing workflow for preparing real-world housing data for machine learning regression models. Professional dashboards, engineered features, statistical visualizations, and reusable preprocessing functions were developed to transform raw housing data into a structured ML-ready dataset.

> *"Good machine learning models are built on strong feature engineering and clean data."*

---

## 📂 Repository Structure

```text
AIML-Internship-Week3-UsmanAsif/
│
├── Week3_HousePrice_Analysis.ipynb
├── train.csv
├── week3_dashboard.png
├── week3_fe_pipeline.png
├── w3_saleprice_distribution.png
├── w3_grlivarea_distribution.png
├── w3_multivariable_scatter.png
├── w3_time_trend.png
├── w3_neighborhood_boxplot.png
├── w3_heatmap.png
├── w3_pairplot.png
├── w3_facetgrid.png
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Kaggle — House Prices: Advanced Regression Techniques |
| Rows | 1,460 |
| Columns | 81 |
| Target | `SalePrice` |
| Dataset Type | Regression |
| Domain | Real Estate |

---

## 🔑 5 Key Findings

### 1. OverallQual is the strongest predictor of SalePrice.
Homes with higher material and construction quality consistently achieved significantly higher selling prices.

### 2. Total living area strongly influences house prices.
Features such as `GrLivArea` and engineered `TotalSF` showed very strong positive correlations with SalePrice.

### 3. Neighborhood significantly affects property value.
Premium neighborhoods consistently showed much higher median prices compared to lower-demand areas.

### 4. Feature engineering improved predictive relationships.
Engineered features such as `TotalSF`, `QualCond`, and `HouseAge` outperformed many original dataset features.

### 5. Most numerical features were highly skewed.
Log transformation successfully reduced skewness and improved distribution normality for regression modeling.

---

## 🛠️ Top 3 Engineered Features

### 1. `TotalSF` — Total Property Area

```python
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
```

Correlation with SalePrice: **~0.78**

This feature combines all usable living space into a single variable, giving the model a stronger representation of total house size.

---

### 2. `QualCond` — Quality × Condition Interaction

```python
df['QualCond'] = df['OverallQual'] * df['OverallCond']
```

This interaction feature captures the combined effect of house quality and condition more effectively than treating them separately.

---

### 3. `HouseAge` — Age of House at Sale Time

```python
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
```

This feature captures depreciation and buyer preference for newer houses, which strongly influences property prices.

---

## 🔧 Tools & Libraries

| Library | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data cleaning & feature engineering |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Encoding & scaling |
| SciPy | Statistical transformations |
| Google Colab | Development environment |

---

## 📋 Notebook Structure (18 Steps)

<details>
<summary><b>Part A — Matplotlib & Seaborn Visualization (Steps 1–6)</b></summary>

- Environment setup & dataset loading
- Distribution analysis
- Multi-variable scatter plots
- Time-based trend analysis
- Correlation heatmaps
- Pair plots & FacetGrid visualizations

</details>

<details>
<summary><b>Part B — Feature Engineering & Encoding (Steps 7–11)</b></summary>

- Creation of 8 engineered features
- Correlation analysis
- Categorical encoding strategy
- Label, One-Hot & Frequency Encoding
- Feature scaling comparison

</details>

<details>
<summary><b>Part C — Skewness Treatment & Feature Selection (Steps 12–15)</b></summary>

- Skewness detection
- log1p, sqrt & Box-Cox transformations
- Variance threshold filtering
- Multicollinearity removal
- Reusable preprocessing functions

</details>

<details>
<summary><b>Part D — Dashboard & Final Report (Steps 16–18)</b></summary>

- Professional 6-chart dashboard
- Feature engineering infographic
- Written analysis report

</details>

---

## 🖼️ Dashboard Preview

### Final Professional Dashboard

<img width="1587" height="1787" alt="image" src="https://github.com/user-attachments/assets/8bc16936-19cd-4f18-bde9-ad7ec0f2eeb2" />


---

## ▶️ How to Run

### Option 1 — Google Colab

1. Open Google Colab
2. Upload `Week3_HousePrice_Analysis.ipynb`
3. Upload `train.csv`
4. Run all cells

---

### Option 2 — Jupyter Notebook

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
jupyter notebook Week3_HousePrice_Analysis.ipynb
```

---

## 📈 Project Highlights

- Complete 18-step ML preprocessing pipeline
- Professional visualization dashboard
- Advanced feature engineering workflow
- Multiple encoding strategies
- Scaling comparison analysis
- Statistical skewness treatment
- Feature selection optimization

---

## 👤 Author

**Usman Asif**  
AI/ML Internship Cohort — Digitech Offerings  
Week 3 of 8 | 10th May, 2026

---

<div align="center">

*Part of the 8-week AI/ML Internship Program by Digitech Offerings*  
*Instructor: Zain Ul Abideen*

</div>

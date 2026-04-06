# 📈 Linear Regression Analysis — Predicting Car Efficiency, Sales & Crime Rates

> **Skills demonstrated:** Linear Regression · Multiple Regression · Feature Engineering · Regularization · Collinearity Detection · Outlier Analysis · Python · Statsmodels · Scikit-learn

---

## 🎯 Project Overview

This project applies **Simple and Multiple Linear Regression** to three real-world business problems:

1. **Auto Dataset** — Predicting fuel efficiency (mpg) from engine specs
2. **Carseats Dataset** — Predicting retail sales from price, location & geography
3. **Boston Dataset** — Predicting crime rates from neighborhood characteristics

Each analysis follows the full ML pipeline: data exploration → model building → diagnostics → optimization.

---

## 📁 Datasets Used

| Dataset | Source | Size | Target Variable |
|---------|--------|------|----------------|
| **Auto** | Carnegie Mellon StatLib (Real) | 397 rows, 8 features | Miles Per Gallon (mpg) |
| **Carseats** | ISLP Package (Simulated) | 400 rows, 11 features | Sales (units in thousands) |
| **Boston** | U.S. Census Bureau (Real) | 506 rows, 13 features | Crime Rate (crim) |

---

## 🔧 Techniques & Tools Applied

| Technique | Purpose |
|-----------|---------|
| Simple Linear Regression (OLS) | Baseline mpg ~ horsepower model |
| Multiple Linear Regression | Modeling mpg from all predictors |
| Interaction Terms | Capturing synergy between features |
| Log / Sqrt / Polynomial Transforms | Handling non-linear relationships |
| Leverage & Studentized Residuals | Detecting outliers & influential points |
| Variance Inflation Factor (VIF) | Detecting multicollinearity |
| Confidence & Prediction Intervals | Quantifying prediction uncertainty |
| ANOVA F-test | Testing overall model significance |

**Libraries:** `numpy` · `pandas` · `statsmodels` · `matplotlib` · `seaborn` · `ISLP`

---

## 📊 Key Results

### Exercise 8 — MPG Prediction from Horsepower (Simple Linear Regression)

| Metric | Value |
|--------|-------|
| **Intercept (β₀)** | 39.94 |
| **Horsepower Coefficient (β₁)** | -0.158 |
| **R² Score** | **0.606** |
| **Residual Standard Error** | 4.91 |
| **Lack of Fit** | 20.9% |
| **Predicted MPG @ horsepower=98** | **24.47 mpg** |
| **95% Confidence Interval** | [23.97, 24.96] |
| **95% Prediction Interval** | [14.81, 34.12] |

> **Finding:** Strong negative relationship — higher horsepower = lower fuel efficiency. Horsepower alone explains **60.6%** of mpg variance. Outliers detected at observations 320 & 327 (studentized residuals > 3).

---

### Exercise 9 — MPG Prediction from All Features (Multiple Linear Regression)

| Predictor | Coefficient | p-value | Significant? |
|-----------|-------------|---------|--------------|
| Intercept | -17.22 | 0.000 | ✅ |
| Cylinders | -0.493 | 0.128 | ❌ |
| Displacement | 0.020 | 0.008 | ✅ |
| Horsepower | -0.017 | 0.220 | ❌ |
| **Weight** | **-0.0065** | **0.000** | ✅ |
| Acceleration | 0.081 | 0.415 | ❌ |
| **Year** | **0.751** | **0.000** | ✅ |
| **Origin** | **1.426** | **0.000** | ✅ |

| F-statistic | p-value | Interpretation |
|------------|---------|---------------|
| 220.98 | ~0 (1.33e-138) | **Strong evidence** — at least one predictor is significant |

> **Key Finding:** Each additional model year improves fuel efficiency by **0.75 mpg** on average — newer cars are significantly more fuel-efficient. Weight is the strongest negative predictor.

**Significant interactions found:**
- `displacement × origin` (p < 0.001) — engine size effect varies by car origin
- `horsepower × weight` (p < 0.001) — combined engine+mass effect on mpg

---

### Exercise 10 — Car Seat Sales Prediction (Qualitative + Quantitative Predictors)

| Predictor | Coefficient | p-value | Interpretation |
|-----------|-------------|---------|----------------|
| Intercept | 13.04 | 0.000 | Baseline sales |
| **Price** | **-0.0545** | **0.000** | $1 price increase → **55 fewer units sold** |
| Urban[Yes] | -0.022 | 0.936 | ❌ Not significant |
| **US[Yes]** | **1.201** | **0.000** | US stores sell **~1,200 more units** |

**Reduced Model (Price + US only):**

| Metric | Full Model (3 predictors) | Reduced Model (2 predictors) |
|--------|--------------------------|------------------------------|
| R² | 0.23928 | 0.23926 |
| Predictors | 3 | 2 |
| Recommendation | ❌ | ✅ **Preferred** (simpler, same fit) |

**95% Confidence Intervals (Reduced Model):**

| Coefficient | Lower | Upper |
|-------------|-------|-------|
| Intercept | 11.79 | 14.27 |
| Price | -0.065 | -0.044 |
| US[Yes] | 0.692 | 1.708 |

> **No outliers detected** — zero observations with |studentized residual| > 3.

---

### Exercise 14 — Collinearity Problem (Multicollinearity Analysis)

| Setup | x1-x2 Correlation | β₁ (True=2.0) | β₂ (True=0.3) | x1 significant? | x2 significant? |
|-------|-------------------|---------------|---------------|-----------------|-----------------|
| Joint model (x1+x2) | **0.772** | 1.615 | 0.943 | ✅ p=0.003 | ❌ p=0.259 |
| x1 alone | — | ~2.1 | — | ✅ | — |
| x2 alone | — | — | ~2.9 | — | ✅ |

> **Critical Finding:** When x1 and x2 are both in the model, collinearity inflates standard errors and makes x2 appear insignificant (p=0.259) — even though it IS significant when modeled alone. This is the **Variance Inflation / Collinearity Problem** in action.

---

### Exercise 13 — Noise Impact on Model Reliability

| Noise Level | β₀ CI Width | β₁ CI Width | Model Quality |
|-------------|-------------|-------------|---------------|
| Low noise (σ=0.01) | **0.004** | **0.005** | 🟢 Very tight CIs |
| Original (σ=0.25) | 0.093 | 0.105 | 🟡 Good CIs |
| High noise (σ=10) | **4.26** | **4.80** | 🔴 Wide, unreliable CIs |

> **Finding:** Noise directly controls how reliably we can estimate coefficients. Low-noise data produces CIs 1000x narrower than high-noise data.

---

## 💡 Business Insights

1. **Retail Pricing Strategy:** A $1 price increase in car seats reduces sales by ~55 units. US-based stores consistently outperform non-US stores by ~1,200 units — a significant location advantage.

2. **Automotive Efficiency:** Model year is a stronger predictor of fuel efficiency than horsepower — every year of manufacturing improvement adds 0.75 mpg regardless of engine specs.

3. **Data Quality Warning:** When two predictors are highly correlated (r=0.77), individual coefficient estimates become unreliable — a critical consideration in any multi-feature business model.

---

## 🗂️ File Structure

```
Chapter_3_Applied_Exercise_Solution/
│
├── Chapter_3.ipynb          ← Main analysis notebook (all exercises)
├── chapter_3.html           ← Rendered HTML version (easy viewing)
├── chapter_3.qmd            ← Quarto source file
└── README.md                ← This file
```

---

## ▶️ How to Run

```bash
# Install dependencies
pip install ISLP statsmodels pandas numpy matplotlib seaborn

# Launch notebook
jupyter notebook Chapter_3.ipynb
```

---

## 📚 Reference

James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023).  
*An Introduction to Statistical Learning with Applications in Python.* Springer.  
Chapter 3: Linear Regression — Applied Exercises 8–15.

---

## 🙏 Acknowledgements

Special thanks to **Karim Aboussel Ham** whose repository [ISLP-applied-solutions](https://github.com/KarimABOUSSELHAM) provided useful guidance and reference during the completion of this project.

---

## 👤 About the Author

**Shad Ali Shah**  
🎓 MPhil Economics Student — School of Economics, Quaid-i-Azam University, Islamabad  
💡 Passionate about the intersection of **Economics**, **Data Science**, and **Machine Learning**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shadalishah)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/shadalishah)

---

*Part of the [ML Portfolio](../README.md) by Shad Ali Shah*

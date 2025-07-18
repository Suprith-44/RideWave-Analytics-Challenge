# RideWave Analytics Challenge: Fare Forecasting in Quahog City

## Overview

This project analyzes RideWave's hourly fare data for bikes, autos, and cars in Quahog City (2021-2023). The goal is to build predictive models for fare forecasting and provide actionable insights for dynamic pricing.

---

## Steps & Results

### 1. Data Exploration & Preparation

- **Libraries Used:** `numpy`, `pandas`, `seaborn`, `matplotlib`, `warnings`
- **Data Loading:** Train and test datasets loaded from CSV.
- **EDA:** 
  - Checked for nulls and duplicates (none found).
  - Boxplots and violin plots compared fare distributions across vehicle types.
  - Correlation heatmaps revealed relationships between fares, traffic, surge multipliers, and other features.
  - Time series line plots showed fare trends over time.
  - Outlier analysis quantified fare anomalies for each vehicle type.
  - Histograms visualized fare distributions (bikes and cars are right-skewed; autos have limited data).

**Key Insights:**
- Cars have the highest and most stable fares.
- Bikes have the lowest fares but highest volatility.
- Autos show moderate volatility and fares.

---

### 2. Time Series Characterization

- **Methods:** Holt and Holt-Winters (Exponential Smoothing)
- **Implementation:** Forecasted average fares for each vehicle type.
- **Visualization:** Plotted actual vs. forecasted fares.
- **Results:** 
  - Holt-Winters captured seasonality better than Holt.
  - Cars: Highest, most stable fares.
  - Bikes: Most volatile, growth potential.
  - Autos: Lowest fares, moderate volatility.

**Business Implications:** 
- Focus on cars for stable revenue.
- Use bikes/autos for market expansion and price-sensitive customers.

---

### 3. Advanced Forecasting & Feature Engineering

- **Feature Engineering:**
  - Lag features, rolling statistics, time-based features, interaction terms, cyclical encoding, weather encoding.
- **Models:** SARIMA/SARIMAX for time series, XGBoost for feature importance.
- **Top Influential Features (via XGBoost):**
  - Bikes: Rolling mean (3), hour_cos, lag_3.
  - Autos: Rolling mean (3), lag_2, rolling std (3).
  - Cars: Rolling mean (3), hour_cos, hour.
- **Results:** SARIMA models provided reasonable forecasts and sMAPE scores.

**Interpretation:** 
- Rolling mean and lag features are most predictive.
- Time-based and cyclical features capture fare patterns.

---

### 4. Ensemble Modeling & Pricing Strategy

- **Models Used:**
  - SARIMAX (bikes, with exogenous variables)
  - XGBoost (autos, with engineered features)
  - VAR (cars, capturing interdependencies)
- **Forecasts:** Generated for test data using each model.
- **Submission:** Combined predictions into a CSV (`submission.csv`) with columns: `timestamp`, `average_fare_bike`, `average_fare_auto`, `average_fare_car`.

**Ensemble Approach:**
- Combine model predictions using weighted averages or stacking (meta-model).
- Weights based on validation accuracy (e.g., MAE, RMSE).
- Ensemble improves robustness and reliability of fare forecasts.

---

## Evaluation Metric

- **Symmetric Mean Absolute Percentage Error (SMAPE):**
  ```python
  smape = np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
  ```
- Lower SMAPE = better performance.

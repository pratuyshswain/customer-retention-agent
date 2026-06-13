# Customer Churn Prediction — Model Evaluation Report

## Best Model: **Logistic Regression**

---

## Metric Comparison

| Metric | Logistic Regression | Random Forest |
|--------|:-------------------:|:-------------:|
| ACCURACY | 0.8162 | 0.8353 ✅ |
| PRECISION | 0.6131 | 0.6659 ✅ |
| RECALL | 0.8284 ✅ | 0.7587 |
| F1 | 0.7047 | 0.7093 ✅ |
| ROC_AUC | 0.8883 ✅ | 0.8881 |

## Top 10 Feature Importances

| Rank | Feature | Importance |
|:----:|---------|:----------:|
| 1 | Contract_Two year | 3.1899 |
| 2 | PaymentMethod_Electronic check | 1.8790 |
| 3 | Contract_One year | 1.8405 |
| 4 | tenure | 1.2244 |
| 5 | InternetService_Fiber optic | 1.0420 |
| 6 | PaperlessBilling | 0.9715 |
| 7 | PaymentMethod_Mailed check | 0.4370 |
| 8 | InternetService_No | 0.3801 |
| 9 | PaymentMethod_Credit card (automatic) | 0.1698 |
| 10 | SeniorCitizen | 0.1209 |

## Performance Targets

| Metric | Target | Achieved | Status |
|--------|:------:|:--------:|:------:|
| ROC_AUC | ≥0.82 | 0.8883 | ✅ PASS |
| F1 | ≥0.60 | 0.7047 | ✅ PASS |
| RECALL | ≥0.70 | 0.8284 | ✅ PASS |
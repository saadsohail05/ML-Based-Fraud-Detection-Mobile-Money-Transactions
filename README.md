# ML-Based Fraud Detection for Mobile Money Transactions

## Project Overview
This project implements machine learning models to detect fraudulent transactions in mobile money transfers. It uses GPU-accelerated implementations for better performance and handles real-world transaction data with features like amount, transaction type, and account balances.

## Features
- GPU-accelerated machine learning models using RAPIDS cuML
- Comprehensive exploratory data analysis (EDA)
- Multiple ML models comparison:
  - K-Nearest Neighbors (KNN) with Euclidean and Manhattan distances
  - Support Vector Machines (SVM) with RBF, Linear, and Polynomial kernels
  - Logistic Regression with L1 and L2 regularization
  - Decision Trees
- Advanced data preprocessing including:
  - Outlier detection using Isolation Forest
  - Feature encoding (One-Hot Encoding and Target Encoding)
  - Feature scaling using StandardScaler
  - Handling class imbalance

## Dataset Features
The dataset includes the following features:
- `transactionType`: Type of transaction (e.g., Deposit, Payment, Transfer)
- `amount`: Monetary value of the transaction
- `initiator`: Sender account ID
- `oldBalInitiator`: Sender's balance before transaction
- `newBalInitiator`: Sender's balance after transaction
- `recipient`: Receiver account ID
- `oldBalRecipient`: Receiver's balance before transaction
- `newBalRecipient`: Receiver's balance after transaction
- `isFraud`: Target variable indicating whether the transaction is fraudulent

## Requirements
- Python 3.x
- RAPIDS cuML (GPU acceleration)
- pandas
- numpy
- scikit-learn
- category_encoders
- seaborn
- matplotlib
- joblib

## Project Structure
```
├── Fraud_Detection_in_Mobile_Money_Transactions.py          # Main Notebook
├── Models/                     # Saved model files
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── logistic_regression.pkl
│   └── svm_model.pkl
├── ohe_columns.pkl            # One-hot encoding columns
├── scaler.pkl                # Feature scaler
└── target_encoder.pkl        # Target encoder for categorical variables
```

## Implementation Details

### Data Preprocessing
1. Outlier Detection and Handling
   - Uses Isolation Forest for anomaly detection
   - Careful handling to preserve potentially important fraud indicators

2. Feature Engineering
   - One-hot encoding for transaction types
   - Target encoding for high-cardinality features (initiator, recipient)
   - Standard scaling for numerical features

### Model Implementation
1. **KNN Classification**
   - Implements both Euclidean and Manhattan distance metrics
   - GPU-accelerated using cuML

2. **SVM Classification**
   - Multiple kernel implementations (RBF, Linear, Polynomial)
   - GPU-accelerated using cuML

3. **Logistic Regression**
   - L1 and L2 regularization
   - GPU-optimized solver

4. **Decision Trees**
   - Implements entropy criterion
   - Visualization of tree structure

### Model Evaluation
- Accuracy metrics
- Classification reports
- Confusion matrices
- Class-wise accuracy for fraud/non-fraud cases
- Training and prediction time measurements

## Results
The project provides comprehensive model evaluation metrics for different models. Here are the detailed results:

### Model Performance Comparison

| Model | Accuracy | Training Time (s) | Prediction Time (s) | Fraud Class Accuracy | Non-Fraud Class Accuracy |
|-------|----------|------------------|-------------------|-------------------|----------------------|
| KNN (Euclidean) | 0.9679 | 3.13 | 74.08 | 0.9572 | 0.9799 |
| KNN (Manhattan) | 0.9683 | 0.03 | 189.10 | 0.9574 | 0.9806 |
| SVM (RBF) | 0.9702 | 1133.07 | 73.77 | 0.9522 | 0.9905 |
| SVM (Linear) | 0.9700 | 468.61 | 42.14 | 0.9454 | 0.9977 |
| SVM (Polynomial) | 0.9692 | 1828.71 | 186.86 | 0.9495 | 0.9914 |
| Logistic Regression (L1) | 0.9667 | 23.39 | 0.02 | 0.9536 | 0.9815 |
| Logistic Regression (L2) | 0.9676 | 6.46 | 0.01 | 0.9531 | 0.9840 |
| Decision Tree | 0.9676 | 17.28 | 0.07 | 0.9498 | 0.9992 |

### Detailed Model Analysis

#### KNN with Euclidean Distance
- **Confusion Matrix:**
```
[[377518   7737]
 [ 18621 416665]]
```
- **Classification Report:**
```
              precision    recall  f1-score   support
       False       0.95      0.98      0.97    385255
        True       0.98      0.96      0.97    435286
    accuracy                           0.97    820541
   macro avg       0.97      0.97      0.97    820541
weighted avg       0.97      0.97      0.97    820541
```

#### KNN with Manhattan Distance
- **Confusion Matrix:**
```
[[377783   7472]
 [ 18544 416742]]
```
- **Classification Report:**
```
              precision    recall  f1-score   support
       False       0.95      0.98      0.97    385255
        True       0.98      0.96      0.97    435286
    accuracy                           0.97    820541
   macro avg       0.97      0.97      0.97    820541
weighted avg       0.97      0.97      0.97    820541
```

#### SVM with RBF Kernel
- **Confusion Matrix:**
```
[[381600   3655]
 [ 20800 414486]]
```
- **Classification Report:**
```
              precision    recall  f1-score   support
       False       0.95      0.99      0.97    385255
        True       0.99      0.95      0.97    435286
    accuracy                           0.97    820541
   macro avg       0.97      0.97      0.97    820541
weighted avg       0.97      0.97      0.97    820541
```

### Key Findings
1. **Best Overall Performance:**
   - SVM with RBF kernel achieved the highest accuracy (0.9702)
   - Best balance between fraud and non-fraud detection

2. **Computational Efficiency:**
   - Logistic Regression (L2) had the fastest prediction time (0.01s)
   - KNN with Manhattan distance had the fastest training time (0.03s)

3. **Model Trade-offs:**
   - SVM models show high accuracy but require significant training time
   - KNN models offer good accuracy with moderate computational requirements
   - Decision Trees provide excellent non-fraud detection (0.9992) with reasonable training time

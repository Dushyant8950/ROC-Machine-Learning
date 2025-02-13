# ROC-Machine-Learning
This code demonstrates the process of training multiple machine learning models on the Breast Cancer dataset, evaluating their performance using Receiver Operating Characteristic (ROC) curves, and identifying optimal classification thresholds.

**Key Steps:**

1. **Data Loading and Preparation:**
   - The Breast Cancer dataset is loaded using `load_breast_cancer()` from `sklearn.datasets`.
   - Features (`X`) and target labels (`y`) are extracted.
   - Features are standardized using `StandardScaler()` to ensure each feature has a mean of 0 and a standard deviation of 1, which is beneficial for models sensitive to feature scaling.
   - The dataset is split into training and testing sets with a 70-30 ratio using `train_test_split()`.

2. **Model Training:**
   - A dictionary named `models` is defined, containing instances of various classifiers:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)
     - k-Nearest Neighbors (k-NN)
     - Gradient Boosting
   - Each model is trained on the training data (`X_train`, `y_train`).

3. **Performance Evaluation:**
   - For each trained model, the predicted probabilities for the positive class are obtained on the test set (`X_test`).
   - ROC curves are plotted for each model by computing the False Positive Rate (FPR) and True Positive Rate (TPR) using `roc_curve()` from `sklearn.metrics`.
   - The Area Under the Curve (AUC) is calculated using `auc()` to quantify the model's ability to distinguish between classes.
   - All ROC curves are plotted on a single graph for comparison, with a diagonal red dashed line representing a random classifier's performance.

4. **Threshold Analysis:**
   - For the Random Forest model, an analysis is performed to determine the optimal classification threshold.
   - The optimal threshold is identified as the point where the difference between TPR and FPR is maximized, which balances sensitivity and specificity.

**Notes:**

- The ROC curve is a graphical representation that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
- The AUC provides a single scalar value to compare models; a higher AUC indicates better model performance.
- Identifying the optimal threshold is crucial in applications where the costs of false positives and false negatives differ.

This approach allows for a comprehensive comparison of different classifiers on the same dataset, aiding in selecting the most appropriate model for the task at hand. 

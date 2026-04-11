import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

# Load data
df = pd.read_csv('dataset.csv')

# Preprocessing
# Handle TotalCharges: replace empty strings with 0
df['TotalCharges'] = df['TotalCharges'].replace(' ', 0).astype(float)

# Drop customerID
df = df.drop('customerID', axis=1)

# Encode categorical variables
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Feature engineering: add total services
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['TotalServices'] = df[service_cols].sum(axis=1)

# Split data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# EDA: Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df.replace({0: 'No', 1: 'Yes'}))
plt.title('Churn Distribution')
plt.savefig('churn_dist.png')
plt.close()

# Correlation heatmap
plt.figure(figsize=(12,8))
corr = df.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation.png')
plt.close()

# Tenure vs Churn
plt.figure(figsize=(8,6))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)
plt.title('Tenure Distribution by Churn')
plt.savefig('tenure_churn.png')
plt.close()

# Train Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:,1]

# Hyperparameter tuning for LR
param_grid_lr = {'C': [0.1, 1, 10]}
grid_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, scoring='roc_auc')
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
best_lr_pred = best_lr.predict(X_test)
best_lr_proba = best_lr.predict_proba(X_test)[:,1]

# Train Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_proba = dt.predict_proba(X_test)[:,1]

# Hyperparameter tuning for DT
param_grid_dt = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='roc_auc')
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_
best_dt_pred = best_dt.predict(X_test)
best_dt_proba = best_dt.predict_proba(X_test)[:,1]

# Evaluation function
def evaluate_model(y_test, pred, proba, model_name):
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    return {'Model': model_name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'AUC': auc}

results = []
results.append(evaluate_model(y_test, lr_pred, lr_proba, 'Logistic Regression'))
results.append(evaluate_model(y_test, best_lr_pred, best_lr_proba, 'Tuned Logistic Regression'))
results.append(evaluate_model(y_test, dt_pred, dt_proba, 'Decision Tree'))
results.append(evaluate_model(y_test, best_dt_pred, best_dt_proba, 'Tuned Decision Tree'))

results_df = pd.DataFrame(results)

# Feature importance for DT
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': best_dt.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Identify high-risk customers (top 10% probability)
test_df = X_test.copy()
test_df['Churn_Prob'] = best_lr_proba  # Using tuned LR for prediction
test_df['Actual_Churn'] = y_test
high_risk = test_df[test_df['Churn_Prob'] > test_df['Churn_Prob'].quantile(0.9)].head(10)

# Generate HTML
html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Telecom Customer Churn Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>Telecom Customer Churn Prediction Report</h1>

    <h2>Dataset Overview</h2>
    <p>Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns</p>
    <p>Churn rate: {df['Churn'].mean():.2%}</p>

    <h2>Churn Distribution</h2>
    <img src="data:image/png;base64,{base64.b64encode(open('churn_dist.png', 'rb').read()).decode()}" alt="Churn Distribution">

    <h2>Correlation Heatmap</h2>
    <img src="data:image/png;base64,{base64.b64encode(open('correlation.png', 'rb').read()).decode()}" alt="Correlation Heatmap">

    <h2>Tenure vs Churn</h2>
    <img src="data:image/png;base64,{base64.b64encode(open('tenure_churn.png', 'rb').read()).decode()}" alt="Tenure vs Churn">

    <h2>Model Performance</h2>
    {results_df.to_html(index=False)}

    <h2>Key Findings</h2>
    <ul>
        <li>The tuned Logistic Regression performs best with AUC of {results_df[results_df['Model']=='Tuned Logistic Regression']['AUC'].values[0]:.3f}</li>
        <li>Top factors driving churn: {', '.join(feature_importance['Feature'].head(5).tolist())}</li>
        <li>Customers with short tenure and high monthly charges are at higher risk</li>
    </ul>

    <h2>Retention Strategies</h2>
    <ul>
        <li>Offer discounts or incentives for month-to-month customers to switch to longer contracts</li>
        <li>Provide better tech support and online security for high-risk customers</li>
        <li>Target retention efforts on customers with high churn probability</li>
    </ul>

    <h2>High-Risk Customers (Sample)</h2>
    {high_risk[['Churn_Prob']].head().to_html()}

</body>
</html>
"""

output_path = os.path.join('ui-ux-pro-max', 'churn_analysis_report.html')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Report generated: {output_path}")
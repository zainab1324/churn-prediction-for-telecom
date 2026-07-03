import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

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

# Generate HTML — editorial report styled to match the sampled chart palette
# (steel blue #3274a1 pulled from the seaborn charts themselves; see
# report_preview.png / README.md for the rendered version this produces)

CSS_STYLES = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #d9e3ea;
    --surface: #e7eef2;
    --ink: #142430;
    --ink-dim: #46596a;
    --ink-faint: #7891a1;
    --line: #b7c9d4;
    --blue: #3274a1;
    --blue-deep: #1c4a63;
    --blue-pale: #c3d9e4;
    --red: #b5433a;
  }

  body { background: var(--bg); color: var(--ink); font-family: 'Work Sans', sans-serif; line-height: 1.6; padding: 0 0 100px; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 0 32px; }

  .masthead { padding: 64px 0 36px; border-bottom: 2px solid var(--ink); margin-bottom: 52px; }
  .kicker { font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase; color: var(--blue-deep); display: flex; align-items: center; gap: 10px; margin-bottom: 20px; }
  .kicker::before { content: ""; width: 22px; height: 1px; background: var(--blue-deep); }
  h1.title { font-family: 'Newsreader', serif; font-weight: 500; font-size: 58px; line-height: 1.04; letter-spacing: -0.01em; color: var(--ink); max-width: 780px; }
  h1.title em { font-style: italic; font-weight: 400; color: var(--blue); }
  .dek { margin-top: 16px; font-size: 16px; color: var(--ink-dim); max-width: 620px; }
  .meta-row { margin-top: 28px; display: flex; gap: 28px; flex-wrap: wrap; font-family: 'Space Mono', monospace; font-size: 11.5px; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.05em; }
  .meta-row b { color: var(--ink-dim); font-weight: 700; }

  .section-head { display: flex; align-items: baseline; gap: 14px; margin-bottom: 24px; }
  .section-num { font-family: 'Space Mono', monospace; font-size: 12px; color: var(--blue-deep); border: 1px solid var(--blue); border-radius: 3px; padding: 2px 7px; }
  .section-title { font-family: 'Newsreader', serif; font-weight: 500; font-size: 26px; color: var(--ink); }
  section { margin-bottom: 60px; }

  .kpi-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 4px; overflow: hidden; }
  .kpi { background: var(--surface); padding: 24px 20px; }
  .kpi-label { font-family: 'Space Mono', monospace; font-size: 10.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-faint); margin-bottom: 12px; }
  .kpi-value { font-family: 'Newsreader', serif; font-size: 34px; font-weight: 500; color: var(--ink); }
  .kpi-value.blue { color: var(--blue-deep); }
  .kpi-sub { margin-top: 6px; font-size: 12px; color: var(--ink-faint); }

  .panel-title { font-family: 'Space Mono', monospace; font-size: 11.5px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--blue-deep); margin-bottom: 10px; display: flex; justify-content: space-between; border-bottom: 1px solid var(--line); padding-bottom: 8px; }
  .panel-title span { color: var(--ink-faint); text-transform: none; letter-spacing: 0; font-style: italic; font-family: 'Newsreader', serif; font-size: 13px; }
  .chart-block { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 18px 18px 4px; }
  .chart-grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 24px; }
  .chart-block img { width: 100%; display: block; border-radius: 3px; }
  .full-width-chart { margin-top: 24px; }
  .full-width-chart img { width: 100%; }

  table { width: 100%; border-collapse: collapse; font-family: 'Space Mono', monospace; font-size: 13px; }
  thead th { text-align: left; font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-faint); font-weight: 700; padding: 0 14px 12px; border-bottom: 1px solid var(--ink); }
  tbody td { padding: 14px; border-bottom: 1px solid var(--line); color: var(--ink-dim); }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr.best { background: var(--blue-pale); }
  tbody tr.best td { color: var(--ink); font-weight: 500; }
  tbody tr.best td:first-child { color: var(--blue-deep); position: relative; }
  tbody tr.best td:first-child::before { content: "\\2713"; margin-right: 8px; color: var(--blue-deep); }
  .num { text-align: right; }

  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
  .card-title { font-family: 'Newsreader', serif; font-size: 19px; font-weight: 500; margin-bottom: 16px; color: var(--ink); padding-bottom: 10px; border-bottom: 1px solid var(--line); }
  ul.findings { list-style: none; }
  ul.findings li { padding-left: 20px; position: relative; margin-bottom: 12px; font-size: 14px; color: var(--ink-dim); }
  ul.findings li::before { content: "\\2192"; position: absolute; left: 0; color: var(--blue); font-weight: 700; }
  ol.strategy { list-style: none; counter-reset: strat; }
  ol.strategy li { counter-increment: strat; padding-left: 30px; position: relative; margin-bottom: 16px; font-size: 14px; color: var(--ink-dim); }
  ol.strategy li::before { content: counter(strat); position: absolute; left: 0; top: -1px; font-family: 'Space Mono', monospace; font-size: 11px; color: var(--surface); background: var(--blue); width: 18px; height: 18px; border-radius: 50%; display: flex; align-items: center; justify-content: center; }

  .risk-table td.prob { color: var(--red); font-weight: 700; }

  footer { margin-top: 76px; padding-top: 22px; border-top: 2px solid var(--ink); font-family: 'Space Mono', monospace; font-size: 11px; color: var(--ink-faint); display: flex; justify-content: space-between; }

  @media (max-width: 720px) {
    h1.title { font-size: 38px; }
    .kpi-strip { grid-template-columns: repeat(2, 1fr); }
    .chart-grid, .two-col { grid-template-columns: 1fr; }
  }
"""

# Best model = highest AUC, determined dynamically (not hardcoded to a model name)
best_model_name = results_df.loc[results_df['AUC'].idxmax(), 'Model']
best_model_auc = results_df['AUC'].max()
top_feature = feature_importance.iloc[0]['Feature']
top5_features = ', '.join(feature_importance['Feature'].head(5).tolist())

model_rows = ""
for _, row in results_df.iterrows():
    row_class = ' class="best"' if row['Model'] == best_model_name else ''
    model_rows += f"""
        <tr{row_class}>
          <td>{row['Model']}</td>
          <td class="num">{row['Accuracy']:.4f}</td>
          <td class="num">{row['Precision']:.4f}</td>
          <td class="num">{row['Recall']:.4f}</td>
          <td class="num">{row['AUC']:.4f}</td>
        </tr>"""

high_risk_sample = high_risk[['Churn_Prob']].head()
risk_rows = ""
for idx, row in high_risk_sample.iterrows():
    risk_rows += f"""
        <tr><td>{idx}</td><td class="num prob">{row['Churn_Prob']:.1%}</td></tr>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Telecom Customer Churn — Analysis Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Work+Sans:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>{CSS_STYLES}</style>
</head>
<body>
<div class="wrap">

  <header class="masthead">
    <div class="kicker">Applied Data Science — Retention Analysis</div>
    <h1 class="title">Telecom Customer<br><em>Churn</em> Prediction</h1>
    <p class="dek">Modeling which customers are about to leave, and why — built on {df.shape[0]:,} subscriber records to guide targeted retention spend.</p>
    <div class="meta-row">
      <span><b>Dataset</b> &nbsp;{df.shape[0]:,} rows &middot; {df.shape[1]} columns</span>
      <span><b>Churn Rate</b> &nbsp;{df['Churn'].mean():.2%}</span>
      <span><b>Models</b> &nbsp;Logistic Regression &middot; Decision Tree</span>
      <span><b>Tuning</b> &nbsp;GridSearchCV, 5-fold CV</span>
    </div>
  </header>

  <section>
    <div class="section-head"><span class="section-num">01</span><span class="section-title">At a Glance</span></div>
    <div class="kpi-strip">
      <div class="kpi">
        <div class="kpi-label">Best Model AUC</div>
        <div class="kpi-value blue">{best_model_auc:.3f}</div>
        <div class="kpi-sub">{best_model_name}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Churn Rate</div>
        <div class="kpi-value">{df['Churn'].mean():.1%}</div>
        <div class="kpi-sub">{int(df['Churn'].sum()):,} of {df.shape[0]:,} customers</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Top Predictive Feature</div>
        <div class="kpi-value" style="font-size:23px;">{top_feature}</div>
        <div class="kpi-sub">Highest importance, tuned Decision Tree</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">High-Risk Cohort</div>
        <div class="kpi-value">Top 10%</div>
        <div class="kpi-sub">Flagged by churn probability</div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head"><span class="section-num">02</span><span class="section-title">Exploratory Analysis</span></div>
    <div class="chart-grid">
      <div class="chart-block">
        <div class="panel-title">Churn Distribution <span>Yes / No</span></div>
        <img src="churn_dist.png" alt="Churn distribution">
      </div>
      <div class="chart-block">
        <div class="panel-title">Tenure vs Churn <span>stacked, 30 bins</span></div>
        <img src="tenure_churn.png" alt="Tenure distribution by churn">
      </div>
    </div>
    <div class="chart-block full-width-chart">
      <div class="panel-title">Feature Correlation Heatmap <span>{df.shape[1]} encoded features</span></div>
      <img src="correlation.png" alt="Correlation heatmap">
    </div>
  </section>

  <section>
    <div class="section-head"><span class="section-num">03</span><span class="section-title">Model Performance</span></div>
    <table>
      <thead><tr><th>Model</th><th class="num">Accuracy</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">AUC</th></tr></thead>
      <tbody>{model_rows}
      </tbody>
    </table>
  </section>

  <section>
    <div class="section-head"><span class="section-num">04</span><span class="section-title">Findings &amp; Response</span></div>
    <div class="two-col">
      <div>
        <div class="card-title">Key Findings</div>
        <ul class="findings">
          <li>{best_model_name} performs best, reaching AUC {best_model_auc:.3f}</li>
          <li>Top factors driving churn: {top5_features}</li>
          <li>Customers with short tenure and high monthly charges are at higher risk</li>
        </ul>
      </div>
      <div>
        <div class="card-title">Retention Strategy</div>
        <ol class="strategy">
          <li>Offer discounts or incentives for month-to-month customers to switch to longer contracts</li>
          <li>Provide stronger tech support and online security bundles for high-risk segments</li>
          <li>Prioritize retention outreach on customers ranked in the top decile of churn probability</li>
        </ol>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head"><span class="section-num">05</span><span class="section-title">High-Risk Customers</span></div>
    <div class="chart-block">
      <div class="panel-title">Sample from top decile by predicted churn probability <span>{best_model_name}</span></div>
      <table class="risk-table">
        <thead><tr><th>Customer Index</th><th class="num">Churn Probability</th></tr></thead>
        <tbody>{risk_rows}
        </tbody>
      </table>
    </div>
  </section>

  <footer>
    <span>Telecom Churn Prediction — Applied Project</span>
    <span>Python &middot; scikit-learn &middot; pandas &middot; seaborn</span>
  </footer>

</div>
</body>
</html>
"""

output_path = 'churn_analysis_report.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Report generated: {output_path}")
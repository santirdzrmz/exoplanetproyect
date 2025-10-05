import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib

# Load the data
df = pd.read_csv('cumulative_2025.10.04_17.40.12.csv')

# Preprocessing
df['koi_disposition_class'] = df['koi_disposition'].apply(
    lambda x: 'CONFIRMED' if x == 'CONFIRMED' 
              else ('CANDIDATE' if x == 'CANDIDATE' else 'OTHER')
)

class_map = {'CONFIRMED': 2, 'CANDIDATE': 1, 'OTHER': 0}
df['koi_disposition_encoded'] = df['koi_disposition_class'].map(class_map)
df = df.select_dtypes(exclude=['object'])
df = df.fillna(value=np.nan)
df = df.dropna()

X = df.drop(columns=['koi_disposition_encoded'])
y = df['koi_disposition_encoded']

# VIF calculation
vif = pd.DataFrame()
df_numeric = X.select_dtypes(include=np.number).fillna(X.mean(numeric_only=True))
vif["feature"] = df_numeric.columns
vif["VIF"] = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]

high_vif_cols = vif[vif['VIF'] > 10]['feature'].tolist()
high_vif_cols = [col for col in high_vif_cols if col != 'const']
X = X.drop(columns=high_vif_cols)


from sklearn.preprocessing import PowerTransformer, MinMaxScaler


# 1. Apply Yeo-Johnson to make distribution normal
pt = PowerTransformer(method='yeo-johnson')
X = pd.DataFrame(pt.fit_transform(X), columns=X.columns, index=X.index)

# 2. Scale to [0,1] after normalization
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)


# Train the model with the best parameters from the notebook
best_params = {
    'subsample': 1.0, 
    'reg_lambda': 2, 
    'reg_alpha': 0.1, 
    'n_estimators': 300, 
    'min_child_weight': 1, 
    'max_depth': 4, 
    'learning_rate': 0.15, 
    'gamma': 0.2, 
    'colsample_bytree': 0.8
}

xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    random_state=42,
    eval_metric='mlogloss',
    **best_params
)


# Train the model on the full dataset
xgb_model.fit(X, y)


# Save the model and scalers
xgb_model.save_model("xgboost_model.json")
joblib.dump(pt, 'power_transformer.joblib')
joblib.dump(scaler, 'min_max_scaler.joblib')

print("XGBoost model and scalers trained and saved.")

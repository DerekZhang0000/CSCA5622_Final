import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('./Kaggle_Sirio_Libanes_ICU_Prediction.csv')

"""
    Data Cleaning
"""
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in df.columns:   # Replaces mising values with the mean/mode for feature importance analysis
    if df[column].dtype in ['float64', 'int64']:
        df[column] = df[column].fillna(df[column].mean())
    elif df[column].dtype == 'object':
        df[column] = df[column].fillna(df[column].mode()[0])
        df[column] = label_encoder.fit_transform(df[column])

"""
    Feature Importance Analysis
"""
X = df.drop(columns=['ICU'])    # We are trying to predict which patients will need to be admitted into the ICU
y = df['ICU']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
importances = feature_importances.sort_values(ascending=False)
# print(feature_importances.sort_values(ascending=False)[:50])

most_important_features = {     # Related features (min/max/mean/diff) are not included if a higher importance is already shown
    'RESPIRATORY_RATE_MAX',
    'BLOODPRESSURE_DIASTOLIC_MIN',
    'AGE_PERCENTIL',
    'TEMPERATURE_MEAN',
    'HEART_RATE_MIN',
    'OXYGEN_SATURATION_MIN',
    'LACTATE_MAX',
    'GENDER',
    'AGE_ABOVE65',
    'HTN',
    'IMMUNOCOMPROMISED',
}

df_selected = df[list(most_important_features) + ['ICU']]
df_selected = df_selected.dropna()
X = df_selected.drop(columns=['ICU'])
y = df_selected['ICU']

"""
    Train and Test
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy (No hyperparameter tuning):", accuracy_score(y_test, model.predict(X_test)))

"""
    Hyperparameter Tuning
"""
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier(random_state=42)
# param_grid = {
#     'n_estimators': [100, 200, 500, 1000],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# First round best = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 1000}
# param_grid = {
#     'n_estimators': [800, 1000, 1200],
#     'max_depth': [None],
#     'min_samples_split': [4, 5, 6],
#     'min_samples_leaf': [1]
# }
# Second round best = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 800}
# param_grid = {
#     'n_estimators': [700, 800, 900],
#     'max_depth': [None],
#     'min_samples_split': [5],
#     'min_samples_leaf': [1]
# }
# Third round best = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 700}
param_grid = {
    'n_estimators': [600, 700],
    'max_depth': [None],
    'min_samples_split': [5],
    'min_samples_leaf': [1]
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=0,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)
model = RandomForestClassifier(random_state=42, max_depth=5, min_samples_leaf=1, min_samples_split=5, n_estimators=600)
model.fit(X_train, y_train)
print("Accuracy (With tuning):", accuracy_score(y_test, model.predict(X_test)))

"""
    Using Model
"""
while True:
    features = {     # Related features (min/max/mean/diff) are not included if a higher importance is already shown
        'RESPIRATORY_RATE_MAX' : None,
        'BLOODPRESSURE_DIASTOLIC_MIN' : None,
        'AGE_PERCENTIL' : None,
        'TEMPERATURE_MEAN' : None,
        'HEART_RATE_MIN' : None,
        'OXYGEN_SATURATION_MIN' : None,
        'LACTATE_MAX' : None,
        'GENDER' : None,
        'AGE_ABOVE65' : None,
        'HTN' : None,
        'IMMUNOCOMPROMISED' : None,
    }
    for feature in list(features.keys()):
        features[feature] = input(f'Enter value for {feature}: ')
    
    input_data = pd.DataFrame([features])
    feature_list = list(most_important_features)
    input_data = input_data[feature_list]
    input_data = input_data.apply(pd.to_numeric, errors='coerce')
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    print(f"Prediction (ICU Admission Needed?): {prediction[0]}")
    print(f"Prediction Probabilities (No: {prediction_proba[0][0]:.2f}, Yes: {prediction_proba[0][1]:.2f})")
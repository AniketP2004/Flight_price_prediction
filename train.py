import json
from random import randint
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv('Clean_Dataset.csv')

print(df.columns)
#  Data preprocessing
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('flight', axis=1)
df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0)
df.stops=  df.stops.map({"zero":0,"one":1,"two_or_more":2})


df = df.join(pd.get_dummies(df.airline, prefix='airline', dtype=int)).drop('airline', axis=1)
df = df.join(pd.get_dummies(df.source_city, prefix='source', dtype=int)).drop('source_city', axis=1)
df = df.join(pd.get_dummies(df.destination_city, prefix='destination', dtype=int)).drop('destination_city', axis=1)
df = df.join(pd.get_dummies(df.arrival_time, prefix='arrival', dtype=int)).drop('arrival_time', axis=1)
df = df.join(pd.get_dummies(df.departure_time, prefix='departure', dtype=int)).drop('departure_time', axis=1)

# Spliting the dataset
X, y = df.drop('price', axis=1), df.price

column_list = X.columns.tolist()
with open("Model_metrics/columns.json", 'w') as f:
    json.dump(column_list, f)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X.head())
# print(y.head())

random_forest = {
  'n_estimators': [100, 200, 300, 400],
  'max_depth': [10, 15, 20, 25, None],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 5, 10],
  'max_features': ['sqrt', 'log2']
}

decision_tree= {
  'max_depth': [10, 15, 20, 25, 30, None],
  'min_samples_split': [2, 5, 10, 20],
  'min_samples_leaf': [1, 5, 10, 15]
}

XGboost= {
  'n_estimators': [200, 300, 400, 500],
  'max_depth': [3, 4, 5, 6, 8],
  'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.2],
  'subsample': [0.7, 0.8, 1.0]
}


# Save function 
def save_model_metrics(name, model, y_test, y_pred ):
    joblib.dump(model, f"C:/Users/project/Desktop/flightprediction/Model_metrics/{name}.pkl")

    metrics = {
        'R2 score': r2_score(y_test, y_pred),
        'mean absolute error': mean_absolute_error(y_test, y_pred),
        'mean square error': mean_squared_error(y_test, y_pred)
    }
    print(f"{name} -> {metrics}")

    with open(f"C:/Users/project/Desktop/flightprediction/Model_metrics/{name}.json", 'w') as f:
        json.dump(metrics, f, indent=4)


def cross_validation(name, model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    print(f"{name} CV: {scores.mean():.4f}")


print("Training ML models")
print(df.stops.unique())
lr = LinearRegression()
cross_validation('Linear Regression', lr, X_train, y_train)
lr.fit(X_train, y_train)
save_model_metrics('Linear Regression', lr, y_test, lr.predict(X_test))

rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), param_distributions=random_forest, n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42)
rf_search.fit(X_train, y_train)
rf = rf_search.best_estimator_
print(f"Random forest best params: {rf_search.best_params_}")
save_model_metrics('Random Forest', rf, y_test, rf.predict(X_test))

Dt = DecisionTreeRegressor(random_state=42)
cross_validation('Decision Tree', Dt, X_train, y_train)
Dt.fit(X_train, y_train)
save_model_metrics('Decision Tree', Dt, y_test, Dt.predict(X_test))

xgbt = XGBRegressor(random_state=42, n_jobs=-1)
cross_validation('Xgboost', xgbt, X_train, y_train)
xgbt.fit(X_train, y_train)
save_model_metrics('Xgboost', xgbt, y_test, xgbt.predict(X_test))

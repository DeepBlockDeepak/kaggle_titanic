from src.evaluate_model import evaluate_model
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_rf_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
    model.fit(X_train, y_train)
    
    # model persistence 
    joblib.dump(model, 'models/rf_titanic_model.pkl')
    return model


def random_forest_main(X_train, y_train, X_val, y_val):
    model = train_rf_model(X_train, y_train)
    predictions = evaluate_model(model, X_val, y_val)

    return model, predictions

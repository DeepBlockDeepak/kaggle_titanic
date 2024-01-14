from src.evaluate_model import evaluate_model
from src.random_forest.rf_train import train_rf_model

def random_forest_main(X_train, y_train, X_val, y_val):
    model = train_rf_model(X_train, y_train)
    predictions = evaluate_model(model, X_val, y_val)

    return model, predictions

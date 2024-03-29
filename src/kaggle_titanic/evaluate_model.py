from sklearn.metrics import accuracy_score


def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    model_name = type(model).__name__
    print(f"Model: {model_name}, Accuracy: {accuracy:.2f}")
    return predictions

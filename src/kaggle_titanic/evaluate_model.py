import numpy as np
from sklearn.metrics import accuracy_score


def evaluate_model(model, X_val, y_val):
    """
    Evaluates the model's performance on the validation set.

    Args:
        model: The trained model WITH A PREDICT method.
        X_val: either pd.DataFrame or numpy array
        y_val: either pd.DataFrame or numpy array
    """
    predictions = model.predict(X_val)
    # Ensure predictions and y_val are NumPy arrays for generalizing
    # logic across models that feed pd.Series here.. flattening needed
    predictions = np.asarray(predictions).flatten()
    y_val = np.asarray(y_val).flatten()
    accuracy = accuracy_score(y_val, predictions)
    model_name = type(model).__name__
    print(f"Model: {model_name}, Accuracy: {accuracy:.2f}")
    return predictions

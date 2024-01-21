from sklearn.naive_bayes import GaussianNB
from src.evaluate_model import evaluate_model


from sklearn.metrics import accuracy_score


def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    model_name = type(model).__name__
    print(f"Model: {model_name}, Accuracy: {accuracy:.2f}")
    return predictions


def naive_bayes_main(X_train, y_train, X_val, y_val):
    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = evaluate_model(model, X_val, y_val)
    return model, predictions

from sklearn.naive_bayes import GaussianNB

from kaggle_titanic.evaluate_model import evaluate_model


def naive_bayes_main(X_train, y_train, X_val, y_val):
    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = evaluate_model(model, X_val, y_val)
    return model, predictions

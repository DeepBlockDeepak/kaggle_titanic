import joblib
from sklearn.svm import SVC
from src.evaluate_model import evaluate_model


def train_svm_model(X_train, y_train):
    svm_model = SVC(kernel="rbf", gamma="scale", C=1, probability=True)
    svm_model.fit(X_train, y_train)

    # saving the svm model
    joblib.dump(svm_model, "models/titanic_svm_model.pkl")

    return svm_model


def svm_main(X_train, y_train, X_val, y_val):
    # Train the SVM model
    svm_model = train_svm_model(X_train, y_train)

    # Evaluate the model
    svm_predictions = evaluate_model(svm_model, X_val, y_val)

    return svm_model, svm_predictions

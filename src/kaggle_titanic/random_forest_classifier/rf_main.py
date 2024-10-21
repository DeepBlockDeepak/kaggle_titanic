import joblib
from sklearn.ensemble import RandomForestClassifier

from kaggle_titanic.base_model.base_model import BaseModel
from kaggle_titanic.evaluate_model import evaluate_model


def train_rf_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # model persistence
    joblib.dump(model, "models/rf_titanic_model.pkl")
    return model


def random_forest_main(X_train, y_train, X_val, y_val):
    model = train_rf_model(X_train, y_train)
    predictions = evaluate_model(model, X_val, y_val)

    return model, predictions

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10)
        self.name = 'RandomForestClassifier'

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        # model persistence
        joblib.dump(self.model, "models/rf_titanic_model.pkl")

    # super confused on what to call here
    def predict(self, X):
        return self.model.predict(X)

import joblib
from sklearn.svm import SVC
from kaggle_titanic.base_model.base_model import BaseModel

def svm_main(X_train, y_train, X_val, y_val):
    model = SVMModel()
    model.fit(X_train, y_train)
    predictions = model.evaluate(X_val, y_val)
    return model, predictions

class SVMModel(BaseModel):
    def __init__(self, kernel="rbf", gamma="scale", C=1, probability=True, save_model=False):
        super().__init__(name='SVMClassifier')
        self.model = SVC(kernel=kernel, gamma=gamma, C=C, probability=probability)
        self.save_model = save_model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        if self.save_model:
            joblib.dump(self.model, "models/titanic_svm_model.pkl")

    def predict(self, X):
        return self.model.predict(X)
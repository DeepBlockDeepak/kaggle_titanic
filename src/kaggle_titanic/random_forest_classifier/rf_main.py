from sklearn.ensemble import RandomForestClassifier as SklearnRFC
from kaggle_titanic.base_model.base_model import BaseModel
import joblib

def random_forest_main(X_train, y_train, X_val, y_val):
    model = RandomForestModel()
    model.fit(X_train, y_train)
    predictions = model.evaluate(X_val, y_val)
    return model, predictions

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__(name='sklearnRandomForestClassifier')
        self.model = SklearnRFC(n_estimators=100, max_depth=10)
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        # Model persistence
        joblib.dump(self.model, "models/rf_titanic_model.pkl")
    
    def predict(self, X):
        return self.model.predict(X)

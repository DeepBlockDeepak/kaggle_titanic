from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
    model.fit(X_train, y_train)
    
    # model persistence 
    joblib.dump(model, 'models/titanic_model.pkl')

    return model

# src/feature_selection.py
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


def perform_rfe(X_train, y_train, n_features_to_select=10):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)

    # Returning the ranking of features
    feature_ranking = sorted(zip(rfe.ranking_, X_train.columns))
    return rfe, feature_ranking

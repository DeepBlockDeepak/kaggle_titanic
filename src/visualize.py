from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importances(model, X_train):
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    plt.figure(figsize=(10, 6))
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature Importances')
    # plt.show()
    plt.savefig('outputs/feature_importances')

def plot_confusion_matrix(y_val, predictions):
    cm = confusion_matrix(y_val, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    # plt.show()
    plt.savefig('outputs/confusion_matrix')

def plot_roc_curve(model, X_val, y_val):
    y_val_probs = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_val_probs)
    auc_score = roc_auc_score(y_val, y_val_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # plt.show()
    plt.savefig('outputs/roc_curve')

def plot_survival_probability_histogram(model, X_test):
    test_probs = model.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(10, 6))
    plt.hist(test_probs, bins=20)
    plt.title('Histogram of Predicted Survival Probabilities')
    plt.xlabel('Survival Probability')
    plt.ylabel('Frequency')
    # plt.show()
    plt.savefig('outputs/survival_histogram')

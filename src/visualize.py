from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_feature_importances(model, X_train):
    model_name = type(model).__name__
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
        plt.figure(figsize=(10, 6))
        feature_importances.nlargest(10).plot(kind='barh')
        plt.title(f'{model_name} - Feature Importances')
        plt.savefig(f'outputs/{model_name.lower()}_feature_importances.png')
    else:
        print(f"Feature importances are not available for the model {model_name}.")


def plot_confusion_matrix(model, y_val, predictions):
    cm = confusion_matrix(y_val, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    model_name = type(model).__name__
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig(f'outputs/{model_name.lower()}_confusion_matrix.png')


def plot_roc_curve(model, X_val, y_val):
    model_name = type(model).__name__
    if hasattr(model, "predict_proba"):
        y_val_probs = model.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, y_val_probs)
        auc_score = roc_auc_score(y_val, y_val_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} - ROC Curve (AUC = {auc_score:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.savefig(f'outputs/{model_name.lower()}_roc_curve.png')
    else:
        print(f"ROC curve cannot be plotted for the model {model_name}.")


def plot_survival_probability_histogram(model, X_test):
    model_name = type(model).__name__
    if hasattr(model, "predict_proba"):
        test_probs = model.predict_proba(X_test)[:, 1]
        plt.figure(figsize=(10, 6))
        plt.hist(test_probs, bins=20)
        plt.title(f'{model_name} - Histogram of Predicted Survival Probabilities')
        plt.xlabel('Survival Probability')
        plt.ylabel('Frequency')
        plt.savefig(f'outputs/{model_name.lower()}_survival_histogram.png')
    else:
        print(f"Survival probability histogram cannot be plotted for the model {model_name}.")


def plot_model_accuracies(model_accuracies):
    """
    Plot a histogram comparing the accuracy scores of different models.

    Args:
    model_accuracies (dict): A dictionary with model names as keys and their accuracy scores as values.
    """
    # Create a DataFrame from the dictionary for easier plotting
    accuracies_df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])

    # Plotting
    plt.figure(figsize=(10, 6))
    ax = accuracies_df.plot(kind='bar', x='Model', y='Accuracy', legend=False, color='skyblue')
    plt.title('Comparison of Model Accuracies')
    plt.ylabel('Accuracy Score')
    plt.xlabel('Models')
    plt.xticks(rotation=0)
    plt.ylim(0, 1)  # Assuming accuracy scores are between 0 and 1
    for i in ax.containers:
        ax.bar_label(i,)
    plt.tight_layout()
    plt.savefig('outputs/model_accuracies.png')    
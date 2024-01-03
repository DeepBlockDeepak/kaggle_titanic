import pandas as pd
from sklearn.model_selection import train_test_split
from src.evaluate_model import evaluate_model
from src.preprocess import preprocess_data
from src.train import train_model
from src.visualize import plot_confusion_matrix, plot_feature_importances, plot_roc_curve, plot_survival_probability_histogram


# Load data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


def create_submission_file(test_data, test_predictions):
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions})
    output.to_csv('submission.csv', index=False)
    print("Your submission was successfully saved!")

# Load data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Process data
X_train, X_test = preprocess_data(train_data, test_data)
y_train = train_data["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Train and evaluate model
model = train_model(X_train, y_train)
predictions = evaluate_model(model, X_val, y_val)

# Plot visualizations
plot_feature_importances(model, X_train)
plot_confusion_matrix(y_val, predictions)
plot_roc_curve(model, X_val, y_val)
plot_survival_probability_histogram(model, X_test)

# Prepare submission file
test_predictions = model.predict(X_test)
create_submission_file(test_data, test_predictions)

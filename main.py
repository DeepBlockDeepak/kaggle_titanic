import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Data Preprocessing
def preprocess_data(data):
    """
    Prepare the dataset for training/testing by selecting relevant features,
    handling missing values, and converting categorical data to numerical.

    Args:
        data (DataFrame): The dataset to be processed.

    Returns:
        DataFrame: The processed dataset with numerical features.
    """

    # Define the numeric and categorical features.
    numeric_features = ["Pclass", "SibSp", "Parch", "Fare", "Age"]
    categorical_features = ["Sex", "Embarked"]
    
    # Extract numeric features and handle missing values for these features.
    numeric_data = data[numeric_features]
    imputer = SimpleImputer(strategy='median')
    numeric_data = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_features)
    
    # Extract categorical features. No need to impute missing values yet.
    categorical_data = data[categorical_features]
    
    # One-hot encode the categorical data, which also automatically handles missing values
    # by creating binary columns that are all zeros for rows where the category is missing.
    categorical_data = pd.get_dummies(categorical_data)
    
    # Combine the numeric and categorical data back into a single DataFrame.
    # This assumes that the indexes have not been altered.
    X = pd.concat([numeric_data, categorical_data], axis=1)
    return X



# Process the training data and separate the features (X) and target (y).
X = preprocess_data(train_data)
y = train_data["Survived"]

# Split the data into training and validation sets to be able to evaluate the
# performance of the model on unseen data.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Model Training
# Create a RandomForestClassifier model. This model operates by constructing a
# multitude of decision trees at training time and outputting the class that is
# the mode of the classes (classification) of the individual trees.
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)

# Predictions and Evaluation
# Use the trained model to predict the outcomes of the validation set and
# compare the predictions to the actual outcomes to calculate the accuracy.
predictions = model.predict(X_val)
print(f"Accuracy: {accuracy_score(y_val, predictions):.2f}")

# Preprocess the test data and make predictions for submission.
# This is the data that Kaggle expects as a submission to evaluate on their
# hidden test set.
X_test = preprocess_data(test_data)
test_predictions = model.predict(X_test)

# Prepare the submission file by creating a DataFrame with the required columns
# and saving it as a CSV file.
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

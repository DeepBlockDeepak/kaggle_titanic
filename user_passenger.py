import joblib
import pandas as pd

from src.features import apply_feature_engineering


def collect_user_input():
    # Collecting basic information
    pclass = int(input("Passenger class (1, 2, or 3): "))
    name = input("Full Name: ")
    sex = input("Sex (male or female): ")
    age = float(input("Age: "))
    sibsp = int(input("Number of siblings/spouses aboard: "))
    parch = int(input("Number of parents/children aboard: "))
    fare = float(input("Fare: "))
    cabin = input("Cabin (leave blank if unknown): ")
    embarked = input(
        "Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton): "
    )

    # Creating a DataFrame from user input
    user_data = pd.DataFrame(
        {
            "Pclass": [pclass],
            "Name": [name],
            "Sex": [sex],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Cabin": [cabin if cabin else "Unknown"],
            "Embarked": [embarked],
        }
    )

    return user_data


def preprocess_user_input(user_input):
    # Apply feature engineering
    processed_input = apply_feature_engineering(user_input)

    # One-hot encoding for categorical variables
    processed_input = pd.get_dummies(
        processed_input, columns=["Sex", "Embarked", "Title", "AgeBin", "FareBin"]
    )

    # List of required columns in the order used during training
    required_columns = [
        "Pclass",
        "SibSp",
        "Parch",
        "Fare",
        "Age",
        "FamilySize",
        "IsAlone",
        "HasCabin",
        "Sex_female",
        "Sex_male",
        "Embarked_C",
        "Embarked_Q",
        "Embarked_S",
        "Title_Capt",
        "Title_Col",
        "Title_Countess",
        "Title_Don",
        "Title_Dona",
        "Title_Dr",
        "Title_Jonkheer",
        "Title_Lady",
        "Title_Major",
        "Title_Master",
        "Title_Miss",
        "Title_Mlle",
        "Title_Mme",
        "Title_Mr",
        "Title_Mrs",
        "Title_Ms",
        "Title_Rev",
        "Title_Sir",
        "AgeBin_Child",
        "AgeBin_Teen",
        "AgeBin_Adult",
        "AgeBin_MiddleAged",
        "AgeBin_Senior",
        "FareBin_Low",
        "FareBin_Below_Average",
        "FareBin_Above_Average",
        "FareBin_High",
    ]

    # Add any missing columns with default value 0
    for col in required_columns:
        if col not in processed_input.columns:
            processed_input[col] = 0

    # Reorder columns to match training data
    processed_input = processed_input[required_columns]

    return processed_input


def main():
    user_input = collect_user_input()

    # Preprocess user input
    preprocessed_input = preprocess_user_input(user_input)

    # Load the trained model
    model = joblib.load("models/rf_titanic_model.pkl")

    # Make a prediction
    prediction_probabilities = model.predict_proba(preprocessed_input)

    # Extract the probability of survival (class 1)
    probability_of_survival = prediction_probabilities[0][1]

    print(f"\n____\nProbability of Survival: {probability_of_survival:.2f}")


if __name__ == "__main__":
    main()

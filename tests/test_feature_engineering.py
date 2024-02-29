import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.feature_eng_pipeline.data_preprocessing import full_pipeline
from src.feature_eng_pipeline.feature_transformers import (
    CreateFamilySizeTransformer,
    CreateIsAloneTransformer,
)
from src.features import extract_title


class TestFeatureEngineering(unittest.TestCase):
    def test_extract_title(self):
        # Mock up some data resembling what the function would actually be working with
        data = pd.DataFrame(
            {
                "Name": [
                    "Braund, Mr. Owen Harris",
                    "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
                    "Heikkinen, Miss. Laina",
                    'Duff Gordon, Sir. Cosmo Edmund ("Mr Morgan")',
                    'Homer, Mr. Harry ("Mr E Haven")',
                    "Kirkland, Rev. Charles Leonard",
                    "Stahelin-Maeglin, Dr. Max",
                    "Crosby, Capt. Edward Gifford",
                    "Dean, Master. Bertram Vere",
                ]
            }
        )

        # Apply the function
        transformed_data = extract_title(data)

        # Check that the titles were extracted correctly
        expected_titles = [
            "Mr",
            "Mrs",
            "Miss",
            "Sir",
            "Mr",
            "Rev",
            "Dr",
            "Capt",
            "Master",
        ]
        self.assertListEqual(list(transformed_data["Title"]), expected_titles)

    def test_family_size_and_is_alone_transformers(self):
        data = pd.DataFrame({"SibSp": [1, 0, 2], "Parch": [0, 2, 1]})
        transformer = Pipeline(
            [
                ("create_family_size", CreateFamilySizeTransformer()),
                ("create_is_alone", CreateIsAloneTransformer()),
            ]
        )

        transformed_data = transformer.fit_transform(data)

        expected_data = pd.DataFrame(
            {
                "SibSp": [1, 0, 2],
                "Parch": [0, 2, 1],
                "FamilySize": [1, 2, 4],
                "IsAlone": [1, 0, 0],
            }
        )

        pd.testing.assert_frame_equal(transformed_data, expected_data)

    def test_full_pipeline(self):
        data = pd.DataFrame(
            {
                "Name": ["Name1", "Name2"],
                "SibSp": [1, 0],
                "Parch": [0, 1],
                "Cabin": [np.nan, "C123"],
                "Age": [22, 30],
                "Fare": [7.25, 71.2833],
                "Sex": ["male", "female"],
                "Embarked": ["S", "C"]
                # Add other columns needed for all transformers
            }
        )

        # Assuming `full_pipeline` is the variable holding your entire pipeline
        transformed_data = full_pipeline.fit_transform(data)

        # Now, need to check specific aspects of `transformed_data`
        # For example, need to check the shape, certain values, etc.
        ###### @BUG self.assertEqual(transformed_data.shape[1], expected_number_of_columns)
        # Add more assertions here based on expected transformations

    def test_handling_unseen_categories(self):
        train_data = pd.DataFrame({"Embarked": ["S", "C", "Q"]})

        test_data = pd.DataFrame({"Embarked": ["S", "C", "Q", "Unknown"]})

        # Fit on train data
        transformed_train = full_pipeline.fit_transform(train_data)

        # Transform test data
        transformed_test = full_pipeline.transform(test_data)

        # Check that transforming test data does not raise an error
        # and produces the correct number of columns
        self.assertEqual(transformed_test.shape[1], transformed_train.shape[1])


if __name__ == "__main__":
    unittest.main()

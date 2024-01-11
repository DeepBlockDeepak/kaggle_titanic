import unittest
from src.features import extract_title
import pandas as pd

class TestFeatureEngineering(unittest.TestCase):

    def test_extract_title(self):
        # Mock up some data resembling what the function would actually be working with
        data = pd.DataFrame({
            'Name': [
                'Braund, Mr. Owen Harris',
                'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
                'Heikkinen, Miss. Laina',
                'Duff Gordon, Sir. Cosmo Edmund ("Mr Morgan")',
                'Homer, Mr. Harry ("Mr E Haven")',
                'Kirkland, Rev. Charles Leonard',
                'Stahelin-Maeglin, Dr. Max',
                'Crosby, Capt. Edward Gifford',
                'Dean, Master. Bertram Vere'
            ]
        })
        
        # Apply the function
        transformed_data = extract_title(data)
        
        # Check that the titles were extracted correctly
        expected_titles = ['Mr', 'Mrs', 'Miss', 'Sir', 'Mr', 'Rev', 'Dr', 'Capt', 'Master']
        self.assertListEqual(list(transformed_data['Title']), expected_titles)

if __name__ == '__main__':
    unittest.main()


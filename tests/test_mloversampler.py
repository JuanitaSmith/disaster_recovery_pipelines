"""
Unit tests for src/mloversampler.py.

To run this test, use `python -m unittest test.test_mloversampler`
"""

from src.mloversampler import MLOverSampling
from sqlalchemy import create_engine
import pandas as pd
import unittest
from src import config


class TestMLOverSamplingClass(unittest.TestCase):

    def setUp(self):
        self.ml = MLOverSampling()

        # load data
        engine = create_engine(config.path_database)
        conn = engine.connect()
        df = pd.DataFrame()
        df = pd.read_sql("select * from messages", con=conn, index_col='id')
        self.y = df.iloc[:, 2:]
        self.X = df[['message']]

    def test_initialization(self):
        """ test of class was instantiated correctly with intended defaults """
        self.assertEqual(self.ml.minority_quantile, 0.25, 'incorrect minority quantile')
        self.assertEqual(self.ml.default_ratio, 1, 'incorrect default duplication ratio')

    def test_read(self):
        """ two dataframes should exist with a more than just 1 row """
        self.assertGreater(self.y.shape[0], 1, 'y should have more than 1 row in the dataframe')
        self.assertGreater(self.X.shape[0], 1, 'X should have more than 1 row in the dataframe')
        self.assertGreater(self.y.shape[1], 1, 'y should have more than 1 column in the dataframe')
        self.assertGreaterEqual(self.X.shape[1], 1, 'X should have at least 1 column in the dataframe')

    def test_get_tail_labels(self):
        """ return a list, class 'offer should be a member of this list """
        self.ml.get_tail_labels(self.y)
        # self.assertTrue(self.self.ml.tail_labels.isinstance == list)
        self.assertIn('offer', self.ml.tail_labels, 'class label `offer` is not in tail label list')

    def test_get_sample_ratio(self):
        """ class 'offer should be the most underrated class """
        self.ml.get_sample_ratio(self.y)
        self.assertEqual(self.ml.ratios.index[0], 'offer', 'offer is not the most underrated class')

    def test_minority_oversampling(self):
        """
        Test if records are duplicated correctly

        For the highest ranked label, select the index of the first record where binary indicator == 1
        Compare the number of times this index appear in y vs y_new
        The difference should match the ratio

        Note: this test will only work with the most underrated ratio

        Example:
            If index `99` appear once in X, and 22 times in X_new, and ratio that was calculated is 21, this is correct.

        """

        X_new, y_new, tail_labels = self.ml.minority_oversampling(self.X, self.y)

        for label in self.ml.tail_labels[:1]:
            ratio = int(self.ml.ratios[label])
            original_records = len(self.y[self.y[label] == 1])
            new_records = len(y_new[y_new[label] == 1])
            print('Label {}: Ratio: {}, Original records: {}, New records: {}'.format(label,
                                                                                      ratio,
                                                                                      original_records,
                                                                                      new_records))
            should_duplicate = original_records + (original_records * ratio)
            print('should duplicate {}'.format(should_duplicate))
            self.assertEqual(new_records, should_duplicate, 'Incorrect number of duplicates for label {}'.format(label))

    def test_max_duplication(self):
        """
        Make sure the new X dataset do not exceed the maximum records it should have

        Multiply total number of minority samples with the highest ratio and add the original number of records
        New dataset should not exceed this number of records.
        """

        X_new, y_new, tail_labels = self.ml.minority_oversampling(self.X, self.y)
        max_ratio = self.ml.ratios.iloc[0]
        max_records = (max_ratio * self.ml.minority_samples) + self.X.shape[0]
        message = 'Number of duplicated records too high: max expected {}, {} found'.format(max_records, X_new.shape[0])
        self.assertLessEqual(X_new.shape[0], max_records, msg=message)


if __name__ == '__main__':
    unittest.main()

# tests = TestMLOverSamplingClass()
#
# tests_loaded = unittest.TestLoader().loadTestsFromModule(tests)
#
# unittest.TextTestRunner().run(tests_loaded)

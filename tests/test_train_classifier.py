"""
Unit tests for src/train_classifier.py.

To run this test individually, run command `python -m unittest tests.test_train_classifier` in the terminal
"""


import unittest
import numpy as np
from src import config
from src.train_classifier import load_data, split_data, build_model, FocalBinaryLoss, RANDOM_STATE


class TestTrainClassifierClass(unittest.TestCase):

    def test_load_data(self):
        """ Test of data from SQLite database is loaded correctly and split into features and labels"""
        X, y, category_names = load_data(config.path_database)
        self.assertGreaterEqual(X.shape[0], 2000, 'X dataset should have at least 2000 rows.')
        self.assertEqual(X.shape[1], 2, 'X dataset should have 2 feature columns.')
        self.assertEqual(X.shape[0], y.shape[0], 'Features and labels sets should have the same number of rows.')
        self.assertEqual(len(category_names), 35, 'There should be 35 labels after cleaning')

    def test_split_data(self):
        """ Make sure split into training and testing datasets are working"""
        X, y, category_names = load_data(config.path_database)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, random_state=RANDOM_STATE)
        self.assertGreaterEqual(X_train.shape[0], X.shape[0], 'X_train oversampling did not work')
        self.assertEqual(X_train.shape[0], y_train.shape[0],
                         'Training datasets should have the same number of rows.')
        self.assertEqual(X_train.shape[1], X_test.shape[1],
                         'Feature datasets should have the same number of columns.')
        self.assertEqual(X_test.shape[0], y_test.shape[0],
                         'Testing datasets should have the same number of rows.')
        self.assertEqual(X_test.shape[1], X_test.shape[1],
                         'Testing datasets should have the same number of columns.')


    def test_build_model(self):
        """ Is focal loss function actually working? This test will run for around 5-10 minutes """

        model = build_model()
        focal_loss = FocalBinaryLoss(gamma=10)
        X, y, category_names = load_data(config.path_database)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        loss = focal_loss.focal_binary_cross_entropy(y_pred, np.array(y_test))
        print('Focal loss = {}'.format(loss))

        self.assertEqual(focal_loss.gamma, 10, 'Focal loss gamma initialization = 10 did not work')
        self.assertGreater(loss, 0, 'Focal loss function is not working, loss 0 is returned')

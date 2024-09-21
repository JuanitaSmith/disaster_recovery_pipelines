"""
Unit tests for src/process_data.py.

To run this test, use `python -m unittest test.test_process_data`
"""


import unittest
import pandas as pd
from src import config
from src.process_data import load_data, clean_data


class TestProcessDataClass(unittest.TestCase):

    def setUp(self):
        self.df_messages = pd.DataFrame()
        self.df_categories = pd.DataFrame()

    def test_load_data(self):
        self.df_messages, self.df_categories = load_data(config.path_messages, config.path_categories)
        self.assertGreaterEqual(self.df_messages.shape[0], 1, 'Messages dataset should have at least 1 row')
        self.assertGreaterEqual(self.df_categories.shape[0], 1, 'Categories dataset should have at least 1 row')

    def test_clean_data(self):
        self.df_messages, self.df_categories = load_data(config.path_messages, config.path_categories)
        df_clean = clean_data(self.df_messages, self.df_categories)

        self.assertEqual(df_clean.duplicated().sum(), 0, 'Duplicate records exist')
        self.assertEqual(df_clean.index.duplicated().sum(), 0, 'Duplicate indexes exist')
        self.assertEqual(sum(df_clean.columns == 'original'), 0, 'original column was not dropped')
        self.assertEqual(sum(df_clean.columns == 'child_alone'), 0, 'child_alone column was not dropped')
        self.assertEqual(len(df_clean[df_clean['related'] == 2]), 0, 'related column have values 2')
        self.assertEqual(df_clean["genre"].dtype.name, 'category', 'column genre must be type "category"')
        self.assertEqual(df_clean.shape[1], 37, 'Dataset is expected to have 37 columns')

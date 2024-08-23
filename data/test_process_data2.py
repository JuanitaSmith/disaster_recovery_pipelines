from process_data import main, load_data, clean_data
import sys
import unittest
import pandas as pd


class TestMLOverSamplingClass(unittest.TestCase):

    def setUp(self):
        self.df_messages = pd.DataFrame()
        self.df_categories = pd.DataFrame()
        sys.argv = ['disaster_messages.csv', 'disaster_categories.csv', 'DisasterResponse.db']

    def test_load_data(self):
        # sys.argv = ['disaster_messages.csv', 'disaster_categories.csv', 'DisasterResponse.db']
        messages_filepath, categories_filepath, database_filepath = sys.argv[:]
        self.df_messages, self.df_categories = load_data(messages_filepath, categories_filepath)
        self.assertGreaterEqual(self.df_messages.shape[0], 1, 'Messages dataset should have at least 1 row')
        self.assertGreaterEqual(self.df_categories.shape[0], 1, 'Categories dataset should have at least 1 row')

    def test_clean_data(self):
        messages_filepath, categories_filepath, database_filepath = sys.argv[:]
        df_messages, df_categories = load_data(messages_filepath, categories_filepath)
        df = clean_data(df_messages, df_categories)

        assert (df.duplicated().sum()) == 0
        assert (df.index.duplicated().sum()) == 0
        assert sum(df.columns == 'original') == 0
        assert sum(df.columns == 'child_alone') == 0
        assert len(df[df['related'] == 2]) == 0
        assert df["genre"].dtype.name == 'category'

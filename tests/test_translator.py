"""
Unit tests for src/translator.py.

To run this test, use `python -m unittest test.test_translator`
"""

from src.translator import OpenAITranslator
import unittest
import os

class TestTranslatorClass(unittest.TestCase):

    def setUp(self):
        """" Initialize the unit test """

        # get API key from environment variables
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            openai_api_key = None

        self.translator = OpenAITranslator(openai_api_key=openai_api_key, translation='False', max_text_to_translate=200)


    def test_initialization(self):
        """ test of class was instantiated correctly with intended defaults """
        self.assertEqual(self.translator.end, 200, 'max number of messages are expected to be 200')
        self.assertEqual(self.translator.batch_size, 400, 'batch size are expected to be 400'),
        self.assertEqual(self.translator.translate, False, 'translation is expected to be False')


    def test_load_data(self):
        """ test reading from SQLite database """
        df_messages, df_translations = self.translator.load_data()
        self.assertGreater(len(df_translations), 1000, 'df_translations should have at 1000 rows preloaded')
        self.assertGreater(len(df_messages), 1, 'df_messages should have at least one row')


    def test_json_translations_to_pandas(self):
        """ Test reading json file into pandas dataframe with already processed messages supplied with the project """
        df_translations = self.translator._json_translations_to_pandas()
        self.assertGreater(len(df_translations), 1000, 'json file should have at least 1000 rows')

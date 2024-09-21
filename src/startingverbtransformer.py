import nltk

import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """ Custom transformer which will extract the starting verb of a sentence """

    def __init__(self):

        self.col = 'message'

    def _tokenize(self, text):
        """ Custom basic tokenizer to support starting verb function """

        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

        detected_urls = re.findall(url_regex, text)
        for url in detected_urls:
            text = text.replace(url, "url")

        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

        return clean_tokens

    def _starting_verb(self, text):
        """ Search text for a starting verb and return True or False """
        sentence_list = nltk.sent_tokenize(text)

        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(self._tokenize(sentence))
            if len(pos_tags) > 0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP', 'VBN', 'VBG'] or first_word == 'RT':
                    return True
        return False

    def fit(self, X, y=None):
        """ Given it is a transformer we can return the self """
        return self

    def transform(self, X):
        """
        Create a new boolean feature indicating of a sentence start with a verb or not

        Args:
            X -> dataframe: can have multiple columns
            col -> string: column in the data dataframe that contains the text to be searched

        Returns:
            df -> return a new dataset with a new BOOLEAN column called 'starting_verb'
        """
        X_tagged = X[self.col].apply(self._starting_verb)
        df = pd.DataFrame(X_tagged)
        df.columns = ['starting_verb']
        return df

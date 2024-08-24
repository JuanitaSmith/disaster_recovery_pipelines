import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk.corpus import wordnet





# def _tokenize(text):
#     # print('Tokenizer triggered')
#     # logger.info('Tokenizer triggered')
#
#     # Replace urls starting with 'https' with placeholder
#     url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
#     # replace urls with a common keyword
#     detected_urls = re.findall(url_regex, text)
#     for url in detected_urls:
#         text = text.replace(url, 'url')
#
#     # Replace urls starting with 'www' with placeholder
#     url_regex = 'www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
#     detected_urls = re.findall(url_regex, text)
#     for url in detected_urls:
#         text = text.replace(url, 'url')
#
#         # replace emails with placeholder
#     email_regex = '([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})'
#     detected_emails = re.findall(email_regex, text)
#     for email in detected_emails:
#         text = text.replace(email, 'email')
#
#         # replace newlines, which can negatively affect performance.
#     text = text.replace("\n", " ")
#     text = text.replace("..", ".")
#     text = text.replace(". .", ".")
#     text = text.replace(" ,.", ".")
#
#     text = re.sub(r'\s+', ' ', text).strip()
#     # text = re.sub(r". ,","",text)
#
#     # normalize text by removing punctuation, remove case and strip spaces
#     text = re.sub(r"[^a-zA-Z0-9]", " ", text)
#     text = text.lower().strip()
#
#     # remove numbers
#     text = re.sub(r'\d+', '', text)
#
#     #   split sentence into words
#     tokens = word_tokenize(text)
#
#     # Remove stopwords, e.g. the, a,
#     tokens = [w for w in tokens if w not in stopwords.words("english")]
#
#     # take words to their core, e.g. children to child, organizations to organization
#     lemmatizer = WordNetLemmatizer()
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok, wordnet.VERB)
#         # ignore tokens that have only 1 character and contains numbers
#         if len(clean_tok) >= 2 & clean_tok.isalpha():
#             clean_tokens.append(clean_tok)
#
#     return clean_tokens

# def tokenize(text):
#
#     # replace urls with a common keyword
#     detected_urls = re.findall(url_regex, text)
#     for url in detected_urls:
#         text = text.replace(url, 'urlplaceholder')
#
#     # normalize text by removing punctuation, remove case and strip spaces
#     text = re.sub(r"[^a-zA-Z0-9]", " ", text)
#     text = text.lower().strip()
#
#     #   split sentence into words
#     tokens = word_tokenize(text)
#
#     # Remove stopwords, e.g. the, a,
#     # tokens = [w for w in tokens if w not in stopwords.words("english")]
#
#     # take words to their core, e.g. children to child
#     lemmatizer = WordNetLemmatizer()
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok)
#         clean_tokens.append(clean_tok)
#
#     return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """ Custom transformer which will extract the starting verb of a sentence """

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

    def transform(self, X, col='message_new'):
        """
        Create a new boolean feature indicating of a sentence start with a verb or not

        Args:
            X -> dataframe: can have multiple columns
            col -> string: column in the data dataframe that contains the text to be searched

        Returns:
            df -> return a new dataset with a new BOOLEAN column called 'starting_verb'
        """
        X_tagged = X[col].apply(self._starting_verb)
        df = pd.DataFrame(X_tagged)
        df.columns = ['starting_verb']
        return df

import sys
from src import config
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from src.utils import *

# save and load models
from pickle import dump

RANDOM_STATE = 88


def load_data(database_filepath):
    # load cleaned messages from SQL database
    engine = create_engine(database_filepath)
    conn = engine.connect()
    df = pd.read_sql('select * from messages', con=conn, index_col='id')

    # sqlite does not keep data types int8 or category as it's not supported, set it again
    df["genre"] = df["genre"].astype("category")
    d = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int8)
    df = df.astype(d)

    # optional: load message translations if available
    df_language = pd.DataFrame()
    try:
        df_language = pd.read_sql('select * from message_language',
                                  con=conn,
                                  index_col='id',
                                  dtype={'is_english': 'boolean'}
                                  )
    finally:
        pass

    # merge dataframes
    if len(df_language) > 0:
        df = df.merge(df_language, on='id', how='left')

    # For messages with no translation or language detection, set is_english = True
    df['is_english'] = np.where(df['is_english'].isnull(), True, df['is_english'])

    # replace message with translation, if message is flagged as not being in English
    df['message_new'] = np.where((df['is_english'] == False) & (~df['translation'].isnull()),
                                 df['translation'],
                                 df['message'])

    X = df[['message_new', 'genre']].copy()
    y = df.iloc[:, 2:-3].copy()
    category_names = y.columns

    return X, y, category_names

def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(config.path_database))

        X, y, category_names = load_data(config.path_database)

        X_train, y_train, X_val, y_val, X_test, y_test = custom_test_train_split(X, y, random_state=RANDOM_STATE)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

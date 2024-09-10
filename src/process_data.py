""" This script import and clean disaster recovery messages """

import sys
import pandas as pd
from sqlalchemy import create_engine
import logging
from src import config

# activate logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename=config.path_log_process_data,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    filemode='w',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def load_data(messages_filepath, categories_filepath):
    """
    Load and return 2 datasets

    Args:
        messages_filepath -> string: file location for messages csv file
        categories_filepath -> string: file location for categories csv file

    Returns:
        messages -> dataframe : pandas dataframe containing all text messages
        categories -> dataframe : pandas dataframe containing all categories text messages belongs to
    """

    logger.info('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
                .format(messages_filepath, categories_filepath))
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))

    # load csv files
    messages = pd.read_csv(messages_filepath)
    logger.info('File {} loaded with shape {}'.format(messages_filepath, messages.shape))

    categories = pd.read_csv(categories_filepath)
    logger.info('File {} loaded with shape {}'.format(categories_filepath, categories.shape))

    return messages, categories


def clean_categories(categories):
    """
    Clean and return categories dataset

    Dataset is loaded with a single column containing all categories e.g. related-1;request-0;offer-0;...
    The content for each column consist of the label name, concatenated with a binary indicator e.g. 'related-1'
    Clean up the dataset to use the label name as the heading, and convert values to contain only binary indicators

    Args:
        categories: pandas dataframe

    Returns:
        categories: cleaned dataframe with column headings and binary indicators
    """

    # create a dataframe splitting the single column into 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True).set_index(categories['id'])

    # extract a list of new column names
    category_column_names = list(categories.iloc[0, :].str.split('-').str[0])
    categories.columns = category_column_names

    # convert category values to contain just binary numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast='integer')

    # drop column child_alone as it has all values = 0
    categories.drop('child_alone', inplace=True, axis=1)

    logger.info('{} categories present after cleaning'.format(categories.shape[1]))
    print('{} categories present after cleaning'.format(categories.shape[1]))

    return categories


def clean_data(messages, categories):
    """
    Cleans and return categories and messages dataset

    Args:
        messages -> dataframe: contain messages texts
        categories -> dataframe: contain categories texts belongs to

    Returns:
        df -> dataframe: cleaned and merged messages and categories
    """

    logger.info('Cleaning data...')
    print('Cleaning data...')

    # CLEAN CATEGORIES
    categories = clean_categories(categories)

    # CLEAN MESSAGES
    # set id as index also in the messages dataset
    messages.set_index('id', inplace=True)

    # merge messages and categories datasets using index
    df = messages.merge(categories, on='id')

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # drop records with duplicated indexes
    df = df[~df.index.duplicated(keep='first')]

    # convert 'genre' to type 'category'
    df["genre"] = df["genre"].astype("category")

    # drop column 'original' as it contains over 50% missing values and are in foreign language
    df.drop('original', inplace=True, axis=1)

    # change 'related' column values with value '2' to '0'
    df['related'] = df['related'].where(df['related'].isin([0, 1]), other=0)

    logger.info('Data cleaning completed with shape {}'.format(df.shape))
    print('Data cleaning completed with shape {}'.format(df.shape))

    return df


def save_data(df, database_filename):
    """
    Stores the clean data into a SQLite database in the specified database file path.

    Args:
        df -> dataframe: cleaned and merged data
        database_filename -> str: database file path
    """

    logger.info('Saving data....')

    engine = create_engine(database_filename)
    logger.info('DB engine: {}'.format(engine))

    try:
        df.to_sql('messages', engine, index=True, if_exists='replace')
        logger.info('Cleaned data saved to database {}'.format(database_filename))
        print('Cleaned data saved to database {}'.format(database_filename))
    except Exception as e:
        logger.error('SQLite update failed with message {}'.format(e))
        print('SQLite update failed with message {}'.format(e))


def main():
    """ main routine """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        df_messages, df_categories = load_data(config.path_messages, config.path_categories)

        df = clean_data(df_messages, df_categories)

        save_data(df, config.path_database)

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

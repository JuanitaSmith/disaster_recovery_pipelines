from process_data import main, load_data, clean_data
import sys

sys.argv = ['disaster_messages.csv', 'disaster_categories.csv', 'DisasterResponse.db']


def test_load_data():

    print(sys.argv)
    messages_filepath, categories_filepath, database_filepath = sys.argv[:]
    df_messages, df_categories = load_data(messages_filepath, categories_filepath)
    assert df_messages.shape[0] > 1
    assert df_categories.shape[0] > 1

    # return df_messages, df_categories


def test_clean_data():

    messages_filepath, categories_filepath, database_filepath = sys.argv[:]
    df_messages, df_categories = load_data(messages_filepath, categories_filepath)
    df = clean_data(df_messages, df_categories)
    assert (df.duplicated().sum()) == 0
    assert (df.index.duplicated().sum()) == 0
    assert sum(df.columns == 'original') == 0
    assert sum(df.columns == 'child_alone') == 0
    assert len(df[df['related'] == 2]) == 0
    assert df["genre"].dtype.name == 'category'

""" helper functions to support ML pipeline """

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import pandas as pd
from sklearn.metrics import classification_report, precision_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import re
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import plot_tree
from src import config

from wordcloud import WordCloud


def evaluate(X, y, model, zero_division=0):
    """Generate model prediction and print model results"""
    y_pred = model.predict(X)

    score = classification_report(y,
                                  y_pred,
                                  target_names=list(y.columns),
                                  zero_division=zero_division,
                                  output_dict=True)

    print(classification_report(y,
                                y_pred,
                                target_names=list(y.columns),
                                zero_division=zero_division,
                                output_dict=False))

    return y_pred, score


def print_results(y, y_pred):
    """ Print scores after cross-validation training """

    roc_auc = roc_auc_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    print('ROC AUC: {}'.format(roc_auc))
    print('Accuracy score: {}'.format(acc))
    return roc_auc


def print_cv_results(cv):
    """ Print cross-validation training results per fold"""

    keys = []
    shapes = []
    examples = []
    for key in list(cv.cv_results_.keys()):
        keys.append(key)
        shapes.append(len(cv.cv_results_.get(key)))
        examples.append(cv.cv_results_.get(key)[0])

    df_results_info = pd.DataFrame({'size': shapes, 'example': examples}, index=keys)
    print(df_results_info)


def cv_plot_scores(cv):
    """ Plot cross validation training vs. testing scores for each fold"""
    nr_params = len(cv.best_params_)
    # print('Nr hyperparameters: {}'.format(nr_params))
    param_list = list(cv.best_params_.keys())
    # print(param_list)
    fig, axes = plt.subplots(nrows=nr_params, ncols=1)

    for i, ax in enumerate(fig.axes):
        # print('Plotting feature {}'.format(i))
        param = param_list[i]
        param_values = list(cv.cv_results_.get('param_' + param))
        mean_test_scores = list(cv.cv_results_.get('mean_test_score'))
        mean_train_scores = list(cv.cv_results_.get('mean_train_score'))

        # replace None with 0
        mean_test_scores = [0 if v is None else v for v in mean_test_scores]
        mean_train_scores = [0 if v is None else v for v in mean_train_scores]

        ax1 = sns.lineplot(x=param_values, y=mean_test_scores, color='green', label='Validation', ax=ax)
        ax1 = sns.lineplot(x=param_values, y=mean_train_scores, color="grey", label='Train', ax=ax1)
        ax1.set_ylabel('Mean Scores')
        ax1.set_xlabel(param)
        ax1.set_xticks(param_values)
        ax1.legend(bbox_to_anchor=(1, 1))
        ax1.set_title('Compare train and validation scores {}'.format(param))

    plt.tight_layout()


def calculate_sample_weights(label_ratio, y, power=1):
    """ Calculate a single sample weight for each row summarizing all labels

    Multi-label sample weights are not supported in the current version
    As a workaround, calculate a sample weight per row across all labels

    Calculation: use the label with the maximum ratio in each row as the sample weight for the entire row

    Args:
        label_ratio: ratio's per label indicating level of imbalance
                     e.g. food with factor 20, related with factor 1
        y: data multi labels
        power: further inflate ratio (ratio ** power)

    Returns:
        weights: dataframe, weight for each row in the target dataset

    """

    # calculate the ratio of each label indicating level of imbalance
    # counts = get_sample_ratio(y)

    # replace class binary indicator with it's weight
    y_copy = y.copy()
    max_ratios = {}
    weights = pd.DataFrame()
    labels = y.columns.to_list()

    for label in labels:
        # print(label, counts[label])
        y_copy[label] = y_copy[label].apply(lambda x: x * label_ratio[label])

    # build a dataframe containing the max weight per row
    for i, row in y_copy[:].iterrows():
        # for each row, get label with maximum ratio
        max_label = row.idxmax()
        # how many times should we copy this row ?
        nr_copies = int(label_ratio[max_label] ** power)
        # # create dictionary of row index and ratio
        max_ratios = {}
        max_ratios[i] = nr_copies
        weight = pd.DataFrame.from_dict(max_ratios, orient='index', columns=['sample_weight'])
        weights = pd.concat([weights, weight], ignore_index=False)

    return weights


def cv_predefined(X_train, y_train, X_val, y_val, pipeline, hyperparameters, label_ratio=None,
                  scoring='roc_auc', verbose=3, error_score=np.nan, calc_sample_ratio=False,
                  random_state=0, nr_splits=3, nr_jobs=None):
    """ Custom cross validation """

    # Build a predefined split with validation set to be used for evaluation

    # Merge training and validation set back together
    # X_train_val = pd.concat([X_train, X_val])
    # y_train_val = pd.concat([y_train, y_val])

    # Create a list where train data indices are -1 and validation data indices are 0
    # split_index = [0 if x in X_val.index else -1 for x in X_train_val.index]
    # pds = PredefinedSplit(test_fold=split_index)
    # print('Number of training splits: {}'.format(pds.get_n_splits()))

    # create grid search object
    gridsearch = GridSearchCV(pipeline,
                              param_grid=hyperparameters,
                              refit=False,
                              return_train_score=True,
                              n_jobs=nr_jobs,
                              scoring=scoring,
                              cv=MultilabelStratifiedKFold(n_splits=nr_splits,
                                                           shuffle=True,
                                                           random_state=random_state),
                              # cv=pds,
                              verbose=verbose,
                              error_score=error_score
                              )

    # compare results of both training and validation datasets
    if calc_sample_ratio:
        sample_weights = calculate_sample_weights(label_ratio, y_train, power=2)
        gridsearch.fit(X_train, y_train, clf__sample_weight=sample_weights)
    else:
        gridsearch.fit(X_train, y_train)

    print('\nBest score: {}'.format(gridsearch.best_score_))
    print("\nBest Parameters:", gridsearch.best_params_)

    return gridsearch


def custom_test_train_split(X, y, random_state=0, embedding=False):
    """ Split data into train, validation and test dataset whilst using stratification """

    # The size of the test set will be 1/K (i.e. 1/n_splits), so you can tune that parameter to control the test size (e.g. n_splits=3 will have test split of size 1/3 = 33% of your data)

    # if embedding:
    #     X.embedding = list(X.embedding.values)

    # STEP 1
    # split data into training and testing dataset
    mskf_1 = MultilabelStratifiedKFold(n_splits=6, shuffle=True, random_state=random_state)

    for train_index, test_index in mskf_1.split(X, y):
        X_train_val, y_train_val = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

    # STEP 2
    # further split training dataset into train and validation datasets
    mskf_2 = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_index, val_index in mskf_2.split(X_train_val, y_train_val):
        X_train, y_train = X_train_val.iloc[train_index], y_train_val.iloc[train_index]
        X_val, y_val = X_train_val.iloc[val_index], y_train_val.iloc[val_index]

    print(
        'Total records: X{}:y{}\n'
        'Train shape: X{}:y{}\n'
        'Validation shape: X{}:y{}\n'
        'Test shape: X{}:y{}'.format(
            X.shape, y.shape, X_train.shape, y_train.shape,
            X_val.shape, y_val.shape, X_test.shape, y_test.shape))

    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_scores(score1, score2, score3, score1_name, score2_name, score3_name):
    """ Compare scores from up to 3 models by category in a bar chart """
    if score1:
        plot1 = pd.DataFrame.from_dict(score1, orient='index', dtype='float16')
        plot1['param'] = score1_name

    if score2:
        plot2 = pd.DataFrame.from_dict(score2, orient='index', dtype='float16')
        plot2['param'] = score2_name
    else:
        plot2 = pd.DataFrame()

    if score3:
        plot3 = pd.DataFrame.from_dict(score3, orient='index', dtype='float16')
        plot3['param'] = score3_name
    else:
        plot3 = pd.DataFrame()

    df_all = pd.concat([plot1, plot2, plot3])
    plt.figure(figsize=(30, 10))
    sns.barplot(data=df_all, x=df_all.index, y='precision', hue='param', width=0.4, dodge=True)
    plt.axhline(0.7)
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Categories', fontsize=18)
    plt.ylabel('Precision score', fontsize=18)
    plt.title('Compare grid search for {}, {}, {}'.format(score1_name, score2_name, score3_name),
              fontsize=18)
    plt.tight_layout()


def plot_label_proportions(y, y_train, y_test):
    """ plot class proportions to analyse imbalances """
    y_train_prop = (y_train.sum() / y_train.shape[0]).sort_values().to_frame()
    y_train_prop.columns = ['train_proportion']
    y_test_prop = (y_test.sum() / y_test.shape[0]).sort_values().to_frame()
    y_test_prop.columns = ['test_proportion']
    y_prop = (y.sum() / y.shape[0]).sort_values().to_frame()
    y_prop.columns = ['before_proportion']
    all = y_train_prop.merge(y_test_prop, left_index=True, right_index=True)
    all = all.merge(y_prop, left_index=True, right_index=True)
    all = all.reindex(y_prop.index)
    all.plot(kind='bar', width=0.6, sharey=True, sharex=True, stacked=False, figsize=(16, 6))
    plt.tight_layout()


def tokenize(text):
    """ Summarize text into words

    Most important functions:
    - Summarize url links starting with http or www to a common phrase 'url
    - Summarize email addresses to a common phrase 'email'
    - Get rid of new lines `\n'
    - Remove all words that are just numbers
    - Remove all words that contain numbers
    - Cleanup basic punctuation like '..', '. .'
    - Remove punctuation
    - Remove words that are just 1 character long after removing punctuation
    - Use lemmatization to bring words to the base

    Args:
        text -> string: Text sentences to be split into words

    Return:
        clean_tokens -> list: List containing most crutial words
    """

    # Replace urls starting with 'https' with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # replace urls with a common keyword
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'url')

    # Replace urls starting with 'www' with placeholder
    url_regex = 'www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'url')

        # replace emails with placeholder
    email_regex = '([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})'
    detected_emails = re.findall(email_regex, text)
    for email in detected_emails:
        text = text.replace(email, 'email')

        # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    text = text.replace("..", ".")
    text = text.replace(". .", ".")
    text = text.replace(" ,.", ".")

    text = re.sub(r'\s+', ' ', text).strip()

    # normalize text by removing punctuation, remove case and strip spaces
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = text.lower().strip()

    # remove numbers
    text = re.sub(r'\d+', '', text)

    #  split sentence into words
    tokens = word_tokenize(text)

    # Remove stopwords, e.g. 'the', 'a',
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # take words to their core, e.g. children to child, organizations to organization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, wordnet.VERB)
        # ignore tokens that have only 1 character or contains numbers
        if len(clean_tok) >= 2 & clean_tok.isalpha():
            clean_tokens.append(clean_tok)

    return clean_tokens


def tokenizer_light(text):
    """ Lighter version of tokenizer function to perform some light text cleaning prior using OPENAI for embeddings

    It's expected that OPENAI are more context-aware, this we should not remove punctuation, stopwords, lemmatize, etc.
    Example there is a big difference between 'I want to help' and 'want help', is openai aware of this difference?

    Most important functions:
    - Summarize url links starting with http or www to a common phrase 'url
    - Summarize email addresses to a common phrase 'email'
    - Get rid of new lines `\n'
    - Remove all words that are just numbers
    - Remove all words that contains numbers
    - Cleanup basic punctuation like '..', '. .'
    - Remove punctuation
    - Remove words that are just 1 character long after removing punctuation
    - Use lemmatization to bring words to the base

    Args:
        text -> string: Text sentences to be split into words

    Return:
        clean_tokens -> list: List containing most crutial words
    """

    # Replace urls starting with 'https' with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # replace urls with a common keyword
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'url')

    # Replace urls starting with 'www' with placeholder
    url_regex = 'www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'url')

        # replace emails with placeholder
    email_regex = '([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})'
    detected_emails = re.findall(email_regex, text)
    for email in detected_emails:
        text = text.replace(email, 'email')

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    # replace basic punctuation errors
    text = text.replace("..", ".")
    text = text.replace(". .", ".")
    text = text.replace(" ,.", ".")

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def word_cloud(counter, title=None, max_words=20):
    """ Plot word cloud with 20 most used words as default """

    f, ax = plt.subplots()
    cloud = WordCloud(
        background_color='white',
        width=2500, height=1800, max_words=max_words,
    ).generate_from_frequencies(frequencies=counter)
    ax.imshow(cloud)
    ax.axis('off')
    plt.title(title, fontsize=30, backgroundcolor='silver')
    plt.show()

def cat_plot(categories, df):
    """produces two wordcloud based on key phrases and
    key adjectives extracted from each neighborhood description
    inputs:
        - neighborhood name (string)
        - list of key phrases (list of strings)
        - list of adjectives (list of strings)
    outputs:
        - two worldclouds
    """

    cv = CountVectorizer(tokenizer=tokenize,
                         token_pattern=None,
                         min_df=3,
                         max_df=0.9,
                         max_features=8000,
                         ngram_range=(1, 3),
                         )

    for cat in categories:
        df_cat = df[df[cat] == 1]

        tokens = cv.fit_transform(df_cat['message'])

        # get word headings
        feature_names = cv.get_feature_names_out()

        freq = pd.DataFrame(tokens.toarray().sum(axis=0))
        freq.columns = ['count']
        freq.index = feature_names
        freq = freq.sort_values(by='count', ascending=False)

        freq_dict = freq['count'].to_dict()
        word_cloud(freq_dict, cat)


def visualize_tree(xgb, bst, tree_to_plot=0):
    """ Visualize tree for XGBoost model

    # gain: the average gain across all splits the feature is used in.
    # weight: the number of times a feature is used to split the data across all trees.
    # cover: the average coverage across all splits the feature is used in.
    # total_gain: the total gain across all splits the feature is used in.
    # total_cover: the total coverage across all splits the feature is used in.
    """

    tree_to_plot = tree_to_plot
    plot_tree(bst, fmap=config.filename_model_featuremap, num_trees=tree_to_plot, rankdir='LR')

    fig = plt.gcf()

    # Get current size
    fig_size = fig.get_size_inches()

    # Set zoom factor
    sizefactor = 20

    # Modify the current size by the factor
    plt.gcf().set_size_inches(sizefactor * fig_size)

    # The plots can be hard to read (known issue). So, separately save it to a PNG, which makes for easier viewing.
    # fig.savefig('tree' + str(tree_to_plot)+'.png')
    plt.show()

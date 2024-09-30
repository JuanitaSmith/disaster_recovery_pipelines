""" helper functions to support ML pipeline training and visualization """

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, cross_val_score
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
from src import config

from wordcloud import WordCloud


def evaluate(X, y, model, zero_division=0):
    """ Generate model prediction and print model results

        Args:
            X: pd.DataFrame, data features
            y: pd.DataFrame, data labels
            model: trained model
            zero_division: {“warn”, 0.0, 1.0, np.nan}, default=”warn”
                Sets the value to return when there is a zero division.
                If set to “warn”, this acts as 0, but warnings are also raised.

        Return:
            y_pred: pd.DataFrame, model predictions
            score: pd.DataFrame, precision/recall scores from function 'classification_report'

    """
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
    """ Print scores after fitting a model

    Precision, recall and F1 scores will be printed by label using scikit-learn `classification_report`.
    Scores will also be calculated for:
    - ROC, using scikit-learn metrics `roc_auc_score`.
    - Overall accuracy: using scikit-learn `accuracy_score`
      Note: label will only be considered as correct, when all 35 classes in a row are correct.
      We can expect a low accuracy between 20-30% generally for multi-label solutions.
    - Mean accuracy by class: using a custom function, calculating the accuracy by label and using the mean

    Args:
        y: pd.DataFrame, actual label values
        y_pred: pd.DataFrame, predicted label values

    Return:
        roc_auc: pd.DataFrame, ROC AUC score
    """

    roc_auc = roc_auc_score(y, y_pred)
    # accuracy_score will only be considered as correct, when all 35 classes in a row are correct.
    acc_global = accuracy_score(y, y_pred)

    # custom accuracy score by class
    acc = (y_pred == y).mean().mean()

    print('ROC AUC: {}'.format(roc_auc))
    print('Accuracy score (by class): {}'.format(acc))
    print('Accuracy score overall: {}'.format(acc_global))
    return roc_auc


def print_cv_results(cv):
    """ Print summary cross-validation training results per fold """

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

    Multi-label sample weights are not well-supported currently.
    As a workaround, calculate a sample weight per row across all labels.

    Calculation: use the label with the maximum ratio in each row as the sample weight for the entire row

    Args:
        label_ratio: Series with index the label name, and value int with ratio.
             Ratio per label indicating level of imbalance
             e.g., label 'food' with factor 20, label 'related' with factor 1
             (Custom class mloversampler.get_sample_ratio() was used in this project)
        y: data multi-labels
        power: further inflate ratio (ratio ** power)

    Returns:
        weights: pd.DataFrame, weight for each row in the target dataset

    """

    # replace class binary indicator with its weight
    y_copy = y.copy()
    weights = pd.DataFrame()
    labels = y.columns.to_list()

    for label in labels:
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
    """ Custom cross validation

    XGBOOST and SCKIT-LEARN together to not support training with train dataset, and validating with validation dataset.
    Below function was experimenting with closing this gap, but was not used for this reason in the end.

    Args:
        X_train: pd.DataFrame, training data features
        y_train: pd.DataFrame, training data labels
        X_val: pd.DataFrame, validation data features
        y_val: pd.DataFrame, validation data labels
        pipeline: sklearn pipeline
        hyperparameters: dict, pipeline hyperparameters
        label_ratio: Series, label ratio (index label, value int ratio)
        scoring: string, cross-validation scoring function
        verbose: integer, verbosity level
        error_score: string, cross-validation error score function
        calc_sample_ratio: boolean, whether to calculate sample weights and use it during cross-validation
        random_state: integer, random state seed
        nr_splits: integer, number of cross-validation splits
        nr_jobs: integer, number of jobs to run in parallel

    Return:
        gridsearch results: dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.
    """

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


def custom_test_val_train_split(X, y, random_state=0, test_split=6, val_split=5):
    """ Split data into train, validation and test dataset whilst using MULTI-LABEL stratification

    The size of the test set will be 1/K (i.e., 1/n_splits).
    Tune this parameter to control the test size.
    (e.g. n_splits=3 will have test split of size 1/3 = 33% of your data)

    Args:
        X: pd.DataFrame, training data features
        y: pd.DataFrame, training data labels
        random_state: integer, random state seed
        test_split: integer, split size for test dataset
        val_split: integer, split size for validation dataset

    Returns:
        X_train: pd.DataFrame, training data features
        y_train: pd.DataFrame, training data labels
        X_val: pd.DataFrame, validation data features
        y_val: pd.DataFrame, validation data labels
        X_test: pd.DataFrame, test data features
        y_test: pd.DataFrame, test data labels
    """

    # STEP 1 - split data into training and testing dataset
    X_train_val, y_train_val, X_test, y_test = custom_test_train_split(X, y,
                                                                       random_state=random_state,
                                                                       test_split=test_split)

    # STEP 2 - further split training dataset into train and validation datasets
    mskf_2 = MultilabelStratifiedKFold(n_splits=val_split, shuffle=True, random_state=random_state)
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


def custom_test_train_split(X, y, random_state=0, test_split=3, silent=True):
    """ Split data into train and test dataset whilst using MULTI-LABEL stratification

    The size of the test set will be 1/K (i.e., 1/n_splits).
    Tune this parameter to control the test size.
    (e.g. n_splits=3 will have test split of size 1/3 = 33% of your data)

    Args:
        X: pd.DataFrame, training data features
        y: pd.DataFrame, training data labels
        random_state: integer, random state seed
        test_split: integer, split size for test dataset
        silent: boolean, if False the splitting results will be printed

    Returns:
        X_train: pd.DataFrame, training data features
        y_train: pd.DataFrame, training data labels
        X_test: pd.DataFrame, test data features
        y_test: pd.DataFrame, test data labels
    """

    # STEP 1 - split data into training and testing dataset
    mskf_1 = MultilabelStratifiedKFold(n_splits=test_split, shuffle=True, random_state=random_state)

    for train_index, test_index in mskf_1.split(X, y):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

    if not silent:
        print(
            'Total records: X{}:y{}\n'
            'Train shape: X{}:y{}\n'
            'Test shape: X{}:y{}'.format(
                X.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    return X_train, y_train, X_test, y_test


def plot_scores(score1, score2, score3, score1_name, score2_name, score3_name, order=None):
    """ Compare scores from up to 3 models by category in a bar chart

    Scores objects are returned using [scikit-learn function 'classification_report']
    (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

    Args:
        score1: str or dict,ext summary of the precision, recall, F1 score for each class for model 1
        score2: str or dict,ext summary of the precision, recall, F1 score for each class for model 2
        score3: : str or dict,ext summary of the precision, recall, F1 score for each class for model 2
        score1_name: str, description of model 1
        score2_name: str, description of model 2
        score3_name: str, description of model 3
        order: list, order class labels should print in, eg in order of imbalance
    """
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
    sns.barplot(data=df_all, x=df_all.index, y='precision', hue='param', width=0.4, dodge=True, order=order)
    plt.axhline(0.7)
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Categories', fontsize=18)
    plt.ylabel('Precision score', fontsize=18)
    plt.title('Compare grid search for {}, {}, {}'.format(score1_name, score2_name, score3_name),
              fontsize=18)
    plt.tight_layout()


def plot_label_proportions(y, y_train, y_test):
    """ Plot class proportions in a bar chart to analyze multi-label class imbalances

    The plot attempts to answer the following questions:
    Does the original, train and test datasets have the same proportions of each class present?
    Did the multi-label stratification split work?

    Args:
        y: pd.DataFrame, full dataset with multi-label labels
        y_train: pd.DataFrame, training dataset after splitting of y -
        y_test: pd.DataFrame, test dataset after splitting of y

    """
    y_train_prop = (y_train.sum() / y_train.shape[0]).sort_values().to_frame()
    y_train_prop.columns = ['train']
    y_test_prop = (y_test.sum() / y_test.shape[0]).sort_values().to_frame()
    y_test_prop.columns = ['test']
    y_prop = (y.sum() / y.shape[0]).sort_values().to_frame()
    y_prop.columns = ['original']
    all_data = y_train_prop.merge(y_test_prop, left_index=True, right_index=True)
    all_data = all_data.merge(y_prop, left_index=True, right_index=True)
    all_data = all_data.reindex(y_prop.index)
    all_data.plot(kind='bar', width=0.6, sharey=True, sharex=True, stacked=False, figsize=(16, 6))
    plt.ylabel('% proportion')
    plt.xlabel('Categories')
    plt.title('Compare proportions of original, train vs test datasets by category')
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
        text: string, Text sentences to be split into words

    Return:
        clean_tokens: list, List containing most crucial words
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
    Example: there is a big difference between 'I want to help' and 'want help', is OpenAI aware of this difference?

    Most important functions:
    - Summarize url links starting with http or www to a common phrase 'url
    - Summarize email addresses to a common phrase 'email'
    - Get rid of new lines `\n'
    - Cleanup basic punctuation like '..', '. .'

    Args:
        text → string: Text sentences to be split into words

    Return:
        clean_tokens → list: List containing most crucial words
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
    """
    Generate a word cloud with the 20 most used words as default.

    Args:
        counter (dict): A dictionary with words as keys and their counts as values.
        title (str, optional): Title of the word cloud. Defaults to None.
        max_words (int): The maximum number of words to generate.

    Return:
        cloud - class WordCloud
    """

    cloud = WordCloud(
        background_color='white',
        width=2500, height=1800, max_words=max_words,
    ).generate_from_frequencies(frequencies=counter)

    return cloud


def cat_word_clouds(categories, df, max_df=0.95, max_features=8000, ngram_range=(1,1)):
    """ Produces word clouds for up to 3 categories

    Args:
        categories (list): List of categories to print word clouds for e.g. ['Food', 'Water', 'Offer']
        df (pd.DataFrame): A pandas dataframe containing texts, with col 'message' containing the text to analyze
        max_df (float in range [0.0, 1.0] or int):
            When building the vocabulary, ignore terms that have a document frequency strictly higher than the given
            threshold (corpus-specific stop words).
            If float, the parameter represents a proportion of documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.
        max_features (int): If not None, build a vocabulary that only considers the top max_features ordered by term
            frequency across the corpus.
            Otherwise, all features are used.
        ngram_range (tuple): Tuple containing the ngram range used to build the vocabulary.
    """

    cv = CountVectorizer(tokenizer=tokenize,
                         token_pattern=None,
                         min_df=3,
                         max_df=max_df,
                         max_features=max_features,
                         ngram_range=ngram_range,
                         )

    fig = plt.figure(figsize=(14, 4))
    for i, cat in enumerate(categories):
        ax = fig.add_subplot(1, 3, i + 1)
        df_cat = df[df[cat] == 1]

        tokens = cv.fit_transform(df_cat['message'])

        # get word headings
        feature_names = cv.get_feature_names_out()

        freq = pd.DataFrame(tokens.toarray().sum(axis=0))
        freq.columns = ['count']
        freq.index = feature_names
        freq = freq.sort_values(by='count', ascending=False)

        freq_dict = freq['count'].to_dict()
        cloud = word_cloud(freq_dict, cat)
        ax.imshow(cloud)
        ax.axis('off')
        ax.set_title(cat)

    plt.suptitle('Word clouds for categories')
    plt.show()


# Do we need to do another grid search when we add an extra feature? Will it then do better with imbalanced labels again?
def objective(trial, features, labels, model, eval_score, n_splits=2, random_state=10):
    """ Grid search using Bayes Optimization with OPTUNA as we can search a lot more parameters faster than GridSearchCV

    Args:
        trial
        features: pd.DataFrame: pandas dataframe containing features (X_train)
        labels: pd.DataFrame: pandas dataframe containing true labels (y_train)
        model: estimator, creating with sci-kit learn Pipeline
        eval_score: scorer callable object created with sci-kit learn make_scorer
        n_splits: int, number of splits for cross-validation
        random_state: int, random seed for splitting

    Return:
        objective_score: score of a trial
    """

    param = {
        'clf__max_delta_step': trial.suggest_float('clf__max_delta_step', low=0, high=100),
        'clf__max_depth': trial.suggest_int('clf__max_depth', low=2, high=12, step=1),
        'clf__reg_lambda': trial.suggest_float('clf__reg_lambda', low=0.0001, high=40),
        'clf__gamma': trial.suggest_float('clf__gamma', low=0.0001, high=50),
        'clf__min_child_weight': trial.suggest_float('clf__min_child_weight', low=1, high=50),
        'clf__reg_alpha': trial.suggest_float('clf__reg_alpha', low=0.0001, high=40),
        'clf__learning_rate': trial.suggest_float('clf__learning_rate', low=0.0001, high=10, log=True),
        'clf__grow_policy': trial.suggest_categorical("clf__grow_policy", ["depthwise", "lossguide"])
    }

    model.set_params(**param)

    objective_score = cross_val_score(estimator=model,
                                      X=features,
                                      y=labels,
                                      scoring=eval_score,
                                      cv=MultilabelStratifiedKFold(n_splits=n_splits,
                                                                   shuffle=True,
                                                                   random_state=random_state),
                                      ).mean()

    return objective_score

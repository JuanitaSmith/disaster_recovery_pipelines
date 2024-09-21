import sys
from sqlalchemy import create_engine
from sklearn.metrics import make_scorer

# save and load models
from pickle import dump

# CUSTOM methods and functions
from src.startingverbtransformer import StartingVerbExtractor
from src.mloversampler import MLOverSampling
from src.focalloss import FocalBinaryLoss
from src.utils import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer, precision_score
import xgboost as xgb
from shutil import rmtree

CACHEDIR = '../cache'
RANDOM_STATE = 10

# activate logging
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename=config.path_log_ml,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    filemode='w',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def load_data(database_filepath):

    print('Loading data...')
    logger.info('Loading data...')

    try:
        # setup connection to SQLITE database
        engine = create_engine(config.path_database)
        conn = engine.connect()
        # load text messages from SQLite database created during ETL pipeline preparation
        sql_statement = 'select * from {}'.format(config.table_messages)
        df_messages = pd.read_sql(sql=sql_statement, con=conn, index_col='id')
    except Exception as error:
        print('Missing or incorrect DATABASE at {} ({})'.format(config.path_database, error))
        logger.error('Missing or incorrect DATABASE at {} ({})'.format(config.path_database, error))
        sys.exit(1)

    print('Number of message records found {}'.format(df_messages.shape[0]))
    logger.info('Number of messages found {}'.format(df_messages.shape[0]))

    # sqlite does not keep data types int8 or category as it's not supported, set it again
    df_messages["genre"] = df_messages["genre"].astype("category")
    d = dict.fromkeys(df_messages.select_dtypes(np.int64).columns, np.int8)
    df = df_messages.astype(d)

    # get a list of category names
    category_names = list(df.select_dtypes(np.int8).columns)
    logger.info(' {} categories found: {}'.format(len(category_names), category_names))

    # split data into features and labels
    X = df[['message', 'genre']].copy()
    y = df.loc[:, category_names].copy()

    print('Shape X: {}'.format(X.shape))
    print('Shape y: {}'.format(y.shape))
    logger.info('Shape X: {}'.format(X.shape))
    logger.info('Shape y: {}'.format(y.shape))

    print(' {} categories found: {}'.format(len(category_names), category_names))
    logger.info(' {} categories found: {}'.format(len(category_names), category_names))

    return X, y, category_names

def build_model():

    # use focal loss to handle imbalance for the evaluation metric
    focal_loss = FocalBinaryLoss(gamma=10)
    fl = make_scorer(focal_loss.focal_binary_cross_entropy)

    params = {
        'objective': 'binary:logitraw',
        'n_jobs': -1,
        'max_depth': 12,
        'min_child_weight': 1.4,
        'gamma': 0.64,
        'eval_metric': fl,
        'booster': 'gbtree',
        'n_estimators': 100,
        'colsample_bytree': 1,
        'subsample': 0.9,
        'tree_method': 'hist',
        'scale_pos_weight': 1,
        'max_delta_step': 15.5,
        'reg_lambda': 0.004,
        'reg_alpha': 0.003,
        'learning_rate': 0.4,
        'verbosity': 2,
        'random_state': RANDOM_STATE,
        'multi_strategy': 'one_output_per_tree',
    }

    text_nlp = Pipeline(steps=[
        ('vect', CountVectorizer(tokenizer=tokenize,
                                 token_pattern=None,
                                 min_df=3,
                                 max_df=0.95,
                                 max_features=10000,
                                 ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(sublinear_tf=False)),
    ],
        verbose=False)

    preprocessor = ColumnTransformer(transformers=[
        ('one_hot', OneHotEncoder(), ['genre']),
        ('starting_verb', StartingVerbExtractor(), ['message']),
        ('text_vect', text_nlp, 'message'),

    ],
        verbose_feature_names_out=False,
        remainder='drop',
        verbose=True,
    )

    pipeline_base = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', xgb.XGBClassifier(**params)),
    ],
        verbose=True,
        # Avoid making continues transformations during cross validation, only the first folds will be transformed
        memory=CACHEDIR,
    )

    return pipeline_base

def split_data(X, y, random_state=RANDOM_STATE):
    """ Data is imbalanced, we need to split data STRATIFIED and oversample it. """

    # split data into train val, and test datasets STRATIFIED
    X_train, y_train, X_val, y_val, X_test, y_test = custom_test_val_train_split(X, y, random_state=random_state)

    # Augment the minority classes
    ml_sampler = MLOverSampling()
    print('Training shapes before augmentation: {} {}'.format(X_train.shape, y_train.shape))
    X_train_sub, y_train_sub, imbalanced_labels = ml_sampler.augment_text(X_train, y_train)
    print('Training shapes after augmentation: {} {}'.format(X_train_sub.shape, y_train_sub.shape))

    return X_train_sub, y_train_sub, X_val, y_val, X_test, y_test


def train_model(model, X, y, random_state=RANDOM_STATE, verbose=1):

    print('Grid search triggered... ')
    scoring = make_scorer(precision_score, average='macro', zero_division=0)

    hyperparameters = {
        "clf__max_depth": [6, 12],
        "clf__n_estimators": [50, 100],
    }

    # hyper parameter tuning
    gridsearch = GridSearchCV(model,
                              param_grid=hyperparameters,
                              refit=False,
                              return_train_score=True,
                              scoring=scoring,
                              cv=MultilabelStratifiedKFold(n_splits=2,
                                                           shuffle=True,
                                                           random_state=random_state),
                              verbose=verbose,
                              )

    gridsearch.fit(X, y)

    print('\nBest score: {}'.format(gridsearch.best_score_))
    print("\nBest Parameters:", gridsearch.best_params_)

    # refit the model with best parameters
    model.set_params(**gridsearch.best_params_)
    model.fit(X, y)

    # clear cache after training
    rmtree(CACHEDIR)

    return model


def evaluate_model(model, X_test, y_test, category_names):
    y_pred, score_base = evaluate(X_test, y_test, model, zero_division=0)
    auc_base = print_results(y_test, y_pred)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(config.path_database))

        X, y, category_names = load_data(config.path_database)


        print('Building model...')
        model = build_model()


        print('Splitting data into train and testing datasets...')
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, random_state=RANDOM_STATE)


        print('Training model with gridsearch...')
        model = train_model(model, X, y)


        print('Evaluating model...')
        evaluate_model(model, X_val, y_val, category_names)


        print('Saving model...\n    MODEL: {}'.format(config.path_model))
        dump(model, open(config.path_model, 'wb'))
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

import os

# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
# import sagemaker
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import time
from src.config import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shutil import rmtree


def make_train_csv(X, y, sagemaker_session, bucket, prefix=prefix, local_path=path_train,
                   filename=filename_train_csv):
    """
    Merges features and labels and converts them into one csv file with labels in the first column.

    File is saved locally and then uploaded to s3. AWS requires no column headings or indexes to be present

    Args:
        X: data features
        y: data labels
        prefix: default S3 sub folder for this project
        local_path: directory where training and validation files will be saved in s3
        filename: name of csv file, ex. 'train.csv'
        sagemaker_session: sagemaker session
        bucket: default bucket assigned to sagemaker session

    Returns: S3 file path where data is stored

    """

    # make data dir, if it does not exist
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # timestamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    full_local_filename = os.path.join(local_path, filename)

    print('Local path: {} with shape {}'.format(full_local_filename, X.shape))

    # save file locally
    df = pd.concat([pd.DataFrame(y), pd.DataFrame(X)], axis=1)
    df.to_csv(full_local_filename, header=False, index=False)

    # will save also the column features an index still ?
    # df.to_parquet(full_local_filename)

    # copy local file to S3
    # s3_path = os.path.join(prefix, local_path)
    # s3_full_path = sagemaker_session.upload_data(path=full_local_filename, bucket=bucket, key_prefix=s3_path)

    # print('File created: {}'.format(s3_full_path))

    # return s3_full_path, df
    return df


def make_test_csv(X, prefix, local_path, filename, sagemaker_session, bucket):
    """
    Saves features to local csv file and upload to S3.

    AWS required that abel column are not present in this file, no column headings or indexes should be present

    Args:
        x: data features
        prefix: default S3 sub folder for this project
        local_path: directory where training and validation files will be saved in s3
        filename: name of csv file, ex. 'train.csv'
        sagemaker_session: sagemaker session
        bucket: default bucket assigned to sagemaker session

    Returns: S3 file path where data is stored

    """

    # make data dir, if it does not exist
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    full_local_filename = os.path.join(local_path, filename)
    print('Local path: {} with shape {}'.format(full_local_filename, X.shape))

    # save file locally
    pd.DataFrame(X).to_csv(full_local_filename, header=False, index=False)
    #     pd.DataFrame(x).to_parquet(full_local_filename)

    # copy local file to S3
    # s3_path = os.path.join(prefix, local_path)
    # s3_full_path = sagemaker_session.upload_data(path=full_local_filename, bucket=bucket, key_prefix=s3_path)

    # print('File created: {}'.format(s3_full_path))

    # return s3_full_path



# def create_feature_map(features):
#     # https://www.kaggle.com/cast42/xgboost-in-python-with-rmspe-v2
#     outfile = open(filename_model_featuremap, 'w')
#     for i, feat in enumerate(features):
#         outfile.write('{0}\t{1}\tq\n'.format(i, feat))
#     outfile.close()


# def train_predict(learner, X_train, y_train, X_test, y_test):
#     """
#     train variance learners to compare performance on accuracy and auc
#
#     args:
#        - learner: the learning algorithm to be trained and predicted on
#        - X_train: features training set
#        - y_train: income training set
#        - X_test: features testing set
#        - y_test: income testing set
#     """
#
#     print('Start of training: {}'.format(learner.__class__.__name__))
#
#     results = {}
#
# #   Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
#     start = time.time() # Get start time
#     learner = learner.fit(X_train, y_train)
#     end = time.time() # Get end time
#
#     # Calculate the training time
#     results['train_time'] = end - start
#
#     # Get the predictions
#     start = time.time() # Get start time
#     predictions_test_proba = learner.predict_proba(X_test)[:, -1]
#     predictions_train_proba = learner.predict_proba(X_train)[:,-1]
#     end = time.time() # Get end time
#
#     predictions_test = [round(num) for num in predictions_test_proba.squeeze()]
#     predictions_train = [round(num) for num in predictions_train_proba.squeeze()]
#
#     # Calculate the total prediction time
#     results['pred_time'] = end - start
#
#     # Compute accuracy on the training samples
#     results['acc_train'] = accuracy_score(predictions_train, y_train)
#
#     # Compute accuracy on test set
#     results['acc_test'] = accuracy_score(predictions_test, y_test)
#
#     # Compute AUC on training set
#     results['auc_train'] = roc_auc_score(y_train, predictions_train_proba, )
#
#     # Compute AUC on testing set
#     results['auc_test'] = roc_auc_score(y_test, predictions_test_proba)
#
#     # Success
#     print("{} training completed".format(learner.__class__.__name__))
#
#     # Return the results
#     return results


def evaluate(X, y, model, zero_division=0):
    """Generate model prediction and print model results"""
    y_pred = model.predict(X)

    print(classification_report(y,
                                y_pred,
                                target_names=list(y.columns),
                                zero_division=zero_division))
    return y_pred


def print_results(y, y_pred, cv):
    """Print best scores and parameters after cross validation training"""

    auc = roc_auc_score(y, y_pred)
    print('AUC aggregate: {}'.format(auc))
    # print('\nBest score: {}'.format(cv.best_score_))
    # print("\nBest Parameters:", cv.best_params_)


def print_cv_results(cv):
    """ Print cross validation training results per fold"""

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
    """ Plot cross validation training and testing scores for each fold"""
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

        ax1 = sns.lineplot(x=param_values, y=mean_test_scores, color='green', label='Validation', ax=ax)
        ax1 = sns.lineplot(x=param_values, y=mean_train_scores, color="grey", label='Train', ax=ax1)
        ax1.set_ylabel('Mean Scores')
        ax1.set_xlabel(param)
        ax1.set_xticks(param_values)
        ax1.legend(bbox_to_anchor=(1,1))
        ax1.set_title('Compare train and validation scores {}'.format(param))

    plt.tight_layout()


def cv_predefined(X_train, y_train, X_val, y_val, pipeline, hyperparameters,
                  scoring='roc_auc', verbose=3, error_score=np.nan):

    # Build a predefined split with validation set to be used for evaluation

    # Merge training and validation set back together
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [0 if x in X_val.index else -1 for x in X_train_val.index]
    pds = PredefinedSplit(test_fold=split_index)
    print('Number of training splits: {}'.format(pds.get_n_splits()))

    # create grid search object
    gridsearch = GridSearchCV(pipeline,
                              param_grid=hyperparameters,
                              refit=False,
                              return_train_score=True,
                              # n_jobs=4,
                              scoring=scoring,
                              # cv=MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=0),
                              cv=pds,
                              verbose=verbose,
                              error_score=error_score
                              )

    # compare results of both training and validation datasets
    gridsearch.fit(X_train_val, y_train_val)

    print('\nBest score: {}'.format(gridsearch.best_score_))
    print("\nBest Parameters:", gridsearch.best_params_)

    # rmtree(cachedir)

    return gridsearch
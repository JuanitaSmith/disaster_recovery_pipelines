import os
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from src.config import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# def make_train_csv(X, y, sagemaker_session, bucket, prefix=prefix, local_path=path_train,
#                    filename=filename_train_csv):
#     """
#     Merges features and labels and converts them into one csv file with labels in the first column.
#
#     File is saved locally and then uploaded to s3. AWS requires no column headings or indexes to be present
#
#     Args:
#         X: data features
#         y: data labels
#         prefix: default S3 sub folder for this project
#         local_path: directory where training and validation files will be saved in s3
#         filename: name of csv file, ex. 'train.csv'
#         sagemaker_session: sagemaker session
#         bucket: default bucket assigned to sagemaker session
#
#     Returns: S3 file path where data is stored
#
#     """
#
#     # make data dir, if it does not exist
#     if not os.path.exists(local_path):
#         os.makedirs(local_path)
#
#     # timestamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
#     full_local_filename = os.path.join(local_path, filename)
#
#     print('Local path: {} with shape {}'.format(full_local_filename, X.shape))
#
#     # save file locally
#     df = pd.concat([pd.DataFrame(y), pd.DataFrame(X)], axis=1)
#     df.to_csv(full_local_filename, header=False, index=False)
#
#     # will save also the column features an index still ?
#     # df.to_parquet(full_local_filename)
#
#     # copy local file to S3
#     # s3_path = os.path.join(prefix, local_path)
#     # s3_full_path = sagemaker_session.upload_data(path=full_local_filename, bucket=bucket, key_prefix=s3_path)
#
#     # print('File created: {}'.format(s3_full_path))
#
#     # return s3_full_path, df
#     return df


# def fpreproc(dtrain, dtest, param):
#     """
#     Preprocessing function used in xgb.cv method
#
#     Recalculate scale_pos_weight for each cross validated sample
#     """
#
#     label = dtrain.get_label()
#     ratio = float(np.sum(label == 0)) / np.sum(label == 1)
#     param["scale_pos_weight"] = ratio
#     return (dtrain, dtest, param)


# def make_test_csv(X, prefix, local_path, filename, sagemaker_session, bucket):
#     """
#     Saves features to local csv file and upload to S3.
#
#     AWS required that abel column are not present in this file, no column headings or indexes should be present
#
#     Args:
#         x: data features
#         prefix: default S3 sub folder for this project
#         local_path: directory where training and validation files will be saved in s3
#         filename: name of csv file, ex. 'train.csv'
#         sagemaker_session: sagemaker session
#         bucket: default bucket assigned to sagemaker session
#
#     Returns: S3 file path where data is stored
#
#     """
#
#     # make data dir, if it does not exist
#     if not os.path.exists(local_path):
#         os.makedirs(local_path)
#
#     full_local_filename = os.path.join(local_path, filename)
#     print('Local path: {} with shape {}'.format(full_local_filename, X.shape))
#
#     # save file locally
#     pd.DataFrame(X).to_csv(full_local_filename, header=False, index=False)
#     #     pd.DataFrame(x).to_parquet(full_local_filename)
#
#     # copy local file to S3
#     # s3_path = os.path.join(prefix, local_path)
#     # s3_full_path = sagemaker_session.upload_data(path=full_local_filename, bucket=bucket, key_prefix=s3_path)
#
#     # print('File created: {}'.format(s3_full_path))
#
#     # return s3_full_path


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


def print_results(y, y_pred, cv):
    """Print best scores and parameters after cross validation training"""

    auc = roc_auc_score(y, y_pred)
    print('AUC aggregate: {}'.format(auc))
    return auc
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
    """ Plot cross validation training vs testing scores for each fold"""
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


# def get_tail_label(y: pd.DataFrame) -> list:
#     """
#     Find the underrepresented targets.
#     Underrepresented targets are those which are observed less than the median occurrence.
#     """
#     # irlbl = y.sum(axis=0)
#     # irlbl = irlbl.max() / irlbl
#     # threshold_irlbl = irlbl.mean()
#     # tail_label = irlbl[irlbl > threshold_irlbl].index.tolist()
#     # print(tail_label)
#     # return tail_label
#
#     # irlbl = y.sum(axis=0)
#     # print(irlbl)
#     # irlbl = irlbl.max() / irlbl
#     # threshold_irlbl = irlbl.mean()
#     # tail_label = irlbl[irlbl > threshold_irlbl].index.tolist()
#     # print(tail_label)
#
#     grand_total = y.sum(axis=0).sum()
#     irlbl = y.sum(axis=0) / grand_total
#     # threshold_irlbl = irlbl.median()
#     threshold_irlbl = irlbl.quantile(0.25)
#     tail_label = irlbl[irlbl < threshold_irlbl].index.tolist()
#
#     return tail_label


# def get_minority_samples(X: pd.DataFrame, y: pd.DataFrame):
#     """
#     Filter datasets containing records with imbalanced targets
#
#     Args:
#         X: data features
#         y: data labels
#
#     Returns:
#         X_sub: pandas.DataFrame, the feature vector minority dataframe
#         y_sub: pandas.DataFrame, the target vector minority dataframe
#
#     """
#     tail_labels = get_tail_label(y)
#     index = y[y[tail_labels].apply(lambda x: (x == 1).any(), axis=1)].index.tolist()
#     X_sub = X.loc[index]
#     y_sub = y.loc[index]
#     print('Imbalanced labels: {}'.format(tail_labels))
#     return X_sub, y_sub, tail_labels


# def get_sample_ratio(df: pd.DataFrame):
#     """
#     Find the underrepresented targets.
#     Underrepresented targets are those which are observed less than the median occurrence.
#     Targets beyond a quantile limit are filtered.
#     Targets which are not under represented are updated with a factor 1, means such rows will not be duplicated
#     Targets which are under represented are updated with factor total mean / target count
#
#     Example
#     'shops' will have factor 20
#     'related' with have factor 1
#
#      Args:
#         df: data labels
#
#     Returns:
#         ratio: Series with label as index, with a ratio/weight column
#
#     """
#
#     tail_labels = get_tail_label(df)
#     full_count = df.sum(axis=0)
#     # print(full_count)
#     ratio = np.where(full_count.index.isin(tail_labels), np.ceil((full_count.mean()) / full_count), 1)
#     ratio = pd.Series(ratio, index=full_count.index)
#     return ratio


# def minority_oversampling(X, y):
#     # calculate the ratio each label should be duplicated to be balanced
#     counts = get_sample_ratio(y)
#
#     # filter datasets with rows that contain imbalanced features
#     X_sub, y_sub, tail_labels = get_minority_samples(X, y)
#     print('Minority samples: {} {}'.format(X_sub.shape, y_sub.shape))
#
#     # replace class binary indicator with it's ratio of duplication
#     y_sub_copy = y_sub.copy()
#     labels = y_sub.columns.to_list()
#     for label in labels:
#         # print(label, counts[label])
#         y_sub_copy[label] = y_sub_copy[label].apply(lambda x: x * counts[label])
#
#     X_list = []
#     y_list = []
#     sample_weight = {}
#
#     for i in range(y_sub.shape[0]):
#         # for each row, get label with maximum ratio
#         max_label = y_sub_copy.iloc[i].idxmax()
#         # how many times should we copy this row ?
#         nr_copies = int(counts[max_label])
#         # get index of row to be copied
#         index = y_sub.iloc[i].name
#         # print(i, max_label, nr_copies, index)
#         # duplicate the rows according to ratio of imbalance
#         for _ in range(nr_copies):
#             y_list.append(y_sub.iloc[i])
#             X_list.append(X_sub.iloc[i])
#
#     X_sub_new = pd.DataFrame(X_list, columns=X.columns.to_list())
#     y_sub_new = pd.DataFrame(y_list, columns=y.columns.to_list())
#
#     X_sub_new = pd.concat([X, X_sub_new])
#     y_sub_new = pd.concat([y, y_sub_new])
#
#     return X_sub_new, y_sub_new, tail_labels


def calculate_sample_weights(label_ratio, y, power=1):
    """ Calculate a single sample weight for each row summarizing all labels

    Multi label sample weights are not supported in the current version
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
    plt.ylabel('Score', fontsize=18)
    plt.title('Precision - Compare grid search for {}, {}, {}'.format(score1_name, score2_name, score3_name),
              fontsize=18)
    plt.tight_layout()
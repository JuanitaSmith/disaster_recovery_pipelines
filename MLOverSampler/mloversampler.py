"""Class to perform over-sampling for multi label classification """

import pandas as pd
import numpy as np


class MLOverSampling:
    """
    Class to perform over-sampling for minority classes in multi label classification

    Reference: Class was inspired by packages MLSMOTE and xxx

    As we are dealing here with NLP, I needed a custom solution
    """

    def __init__(self, minority_quantile=0.25, default_ratio=1):

        self.tail_labels = []
        self.minority_quantile = minority_quantile
        self.default_ratio = default_ratio

    def get_tail_label(self, y: pd.DataFrame):
        """
        Find the underrepresented targets.
        Underrepresented targets are those which mean appear in the 0.25 quantile
        """

        grand_total = y.sum(axis=0).sum()
        irlbl = y.sum(axis=0) / grand_total
        threshold_irlbl = irlbl.quantile(self.minority_quantile)
        self.tail_labels = irlbl[irlbl < threshold_irlbl].index.tolist()

    def get_minority_samples(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Filter datasets to only contain records with imbalanced targets

        Args:
            X: pd.DataFrame, data features
            y: pd.DataFrame, data labels

        Returns:
            X_sub: pd.DataFrame, filtered data features rows
            y_sub: pd.DataFrame, filtered data labels rows

        """
        index = y[y[self.tail_labels].apply(lambda x: (x == 1).any(), axis=1)].index.tolist()
        X_sub = X.loc[index]
        y_sub = y.loc[index]
        print('Imbalanced labels: {}'.format(self.tail_labels))
        return X_sub, y_sub

    def get_sample_ratio(self, df: pd.DataFrame):
        """
        Targets which are not underrepresented are updated with a factor 1
        Targets which are underrepresented are updated with factor total mean / target count

        Example
        - label 'shops' will have factor 20
        - label 'security' will have factor 6
        - label 'related' with have factor 1

         Args:
            df: data labels

        Returns:
            ratio: Series with label as index, with a ratio/weight column
        """

        # tail_labels = self._get_tail_label(df)
        full_count = df.sum(axis=0)
        # print(full_count)
        ratio = np.where(full_count.index.isin(self.tail_labels), np.ceil((full_count.mean()) / full_count),
                         self.default_ratio)
        ratio = pd.Series(ratio, index=full_count.index)
        return ratio

    def minority_oversampling(self, X, y):
        """
        Duplicate each minority samples multiple times according to the label ratio

        Args:
            X: pd.DataFrame, data features
            y: pd.DataFrame, data labels

        Returns:
            X:sub_new: pd.DataFrame, enhanced feature dataset with minority classes duplicated
            y_sub_new: pd.DataFrame, enhanced target dataset with minority classes duplicated
            tail_labels: list, label names that are underrepresented (mean are in the 0.25 percentile)
        """

        # calculate the ratio each label should be duplicated to be balanced
        self.get_sample_ratio(y)

        # filter datasets with rows that contain imbalanced features
        X_sub, y_sub, tail_labels = self.get_minority_samples(X, y)
        print('Minority samples: {} {}'.format(X_sub.shape, y_sub.shape))

        # replace class binary indicator with it's ratio of duplication
        y_sub_copy = y_sub.copy()
        labels = y_sub.columns.to_list()
        for label in labels:
            # print(label, counts[label])
            y_sub_copy[label] = y_sub_copy[label].apply(lambda x: x * self.tail_labels[label])

        X_list = []
        y_list = []
        # sample_weight = {}

        for i in range(y_sub.shape[0]):
            # for each row, get label with maximum ratio
            max_label = y_sub_copy.iloc[i].idxmax()
            # how many times should we copy this row ?
            nr_copies = int(self.tail_labels[max_label])
            # get index of row to be copied
            # index = y_sub.iloc[i].name
            # print(i, max_label, nr_copies, index)
            # duplicate the rows according to ratio of imbalance
            for _ in range(nr_copies):
                y_list.append(y_sub.iloc[i])
                X_list.append(X_sub.iloc[i])

        X_sub_new = pd.DataFrame(X_list, columns=X.columns.to_list())
        y_sub_new = pd.DataFrame(y_list, columns=y.columns.to_list())

        X_sub_new = pd.concat([X, X_sub_new])
        y_sub_new = pd.concat([y, y_sub_new])

        return X_sub_new, y_sub_new, tail_labels
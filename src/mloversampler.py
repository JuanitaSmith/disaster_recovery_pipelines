"""Class to perform over-sampling for multi-label classification """

import pandas as pd
import numpy as np
import nlpaug.augmenter.word as naw


class MLOverSampling:
    """
    Class to perform over-sampling for minority classes in MULTI-LABEL classification problems

    Reference: Class was inspired by packages such as 'MLSMOTE'.
    MLSMOTE needs as input NLP text already converted to a numerical dataset.
    As I needed to use a pipeline for transformation in this project, I could not use a KNN approach, as in the
    input as a text string.

    Two Options are available instead:
    1) Simple copy of the minority classes records multiple times
    2) Use augmentation to generate new similar sentences by replacing some words with synonyms

    Args:
        tail_labels -> list: names of labels that are the minority e.g. ['shops', 'offer']
        minority_quantile -> float: quantile of averages that should be considered as the minority (default 0.25)
        default_ratio -> int: default ratio to give to other classes that are NOT the minority (default 1)
    """

    def __init__(self, minority_quantile=0.25, default_ratio=1):
        """ Initialization of class"""
        self.tail_labels = []
        self.minority_quantile = minority_quantile
        self.default_ratio = default_ratio
        self.ratios = pd.Series()
        self.minority_samples = 0
        self.aug = naw.SynonymAug(aug_src='wordnet', lang='eng')

    def get_tail_labels(self, y: pd.DataFrame):
        """
        Find the underrepresented targets.
        Underrepresented targets are those which mean appear in the 0.25 quantile

        Args:
            y: pd.DataFrame, data labels
        """

        grand_total = y.sum(axis=0).sum()
        label_mean = y.sum(axis=0) / grand_total
        threshold_label_mean = label_mean.quantile(self.minority_quantile)
        self.tail_labels = label_mean[label_mean < threshold_label_mean].index.tolist()
        return self.tail_labels

    def _get_minority_samples(self, X: pd.DataFrame, y: pd.DataFrame):
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
        self.minority_samples = X_sub.shape[0]
        print('Imbalanced labels: {}'.format(self.tail_labels))
        return X_sub, y_sub

    def get_sample_ratio(self, y: pd.DataFrame):
        """
        Targets which are NOT underrepresented are updated with a factor 1
        Targets which are underrepresented are updated with factor total mean / target count

        Example
        - label 'shops' will have factor 20
        - label 'security' will have factor 6
        - label 'related' with have factor 1

         Args:
            y: data labels

        Returns:
            ratio: Series with label as index, with a ratio/weight column
        """

        self.get_tail_labels(y)
        full_count = y.sum(axis=0)
        # print(full_count)
        ratio = np.where(full_count.index.isin(self.tail_labels), np.ceil((full_count.mean()) / full_count),
                         self.default_ratio)
        self.ratios = pd.Series(ratio, index=full_count.index).sort_values(ascending=False)
        return self.ratios

    def minority_oversampling(self, X, y):
        """
        Duplicate minority samples multiple times, according to the label ratio

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

        # filter datasets with rows that only contain imbalanced features set to 1
        X_sub, y_sub = self._get_minority_samples(X, y)
        print('Minority samples: {} {}'.format(X_sub.shape, y_sub.shape))

        # replace class binary indicator with its ratio/weight of duplication
        y_sub_copy = y_sub.copy()
        labels = y_sub.columns.to_list()
        for label in labels:
            y_sub_copy[label] = y_sub_copy[label].apply(lambda x: x * self.ratios[label])

        X_list = []
        y_list = []

        for i in range(y_sub.shape[0]):
            # for each row, get label with maximum ratio
            max_label = y_sub_copy.iloc[i].idxmax()
            # how many times should we copy this row?
            nr_copies = int(self.ratios[max_label])

            # duplicate the rows according to ratio of imbalance
            for _ in range(nr_copies):
                y_list.append(y_sub.iloc[i])
                X_list.append(X_sub.iloc[i])

        X_sub_new = pd.DataFrame(X_list, columns=X.columns.to_list())
        y_sub_new = pd.DataFrame(y_list, columns=y.columns.to_list())

        X_sub_new = pd.concat([X, X_sub_new])
        y_sub_new = pd.concat([y, y_sub_new])

        X_sub_new = X_sub_new.astype(X.dtypes.to_dict())
        y_sub_new = y_sub_new.astype(y.dtypes.to_dict())

        return X_sub_new, y_sub_new, self.tail_labels

    def augment_text(self, X, y):
        """
        Augment minority samples multiple times, according to the label ratio
        Some words will be replaced with synonyms

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

        # filter datasets with rows that only contain imbalanced features set to 1
        X_sub, y_sub = self._get_minority_samples(X, y)
        print('Minority samples: {} {}'.format(X_sub.shape, y_sub.shape))

        # replace class binary indicator with its ratio/weight of duplication
        y_sub_copy = y_sub.copy()
        labels = y_sub.columns.to_list()
        for label in labels:
            y_sub_copy[label] = y_sub_copy[label].apply(lambda x: x * self.ratios[label])

        idx = []
        new_text = []
        genre = []

        for i in range(y_sub.shape[0]):
            # for each row, get label with maximum ratio
            max_label = y_sub_copy.iloc[i].idxmax()
            # how many times should we copy this row?
            nr_copies = int(self.ratios[max_label])

            for index, row in X_sub.iloc[[i]].iterrows():
                augmented_texts = self.aug.augment(row.message, n=nr_copies)
                augmented_texts = [x.replace(" ' ", "'") for x in augmented_texts]
                for text in augmented_texts:
                    new_text.append(text)
                    idx.append(index)
                    genre.append(row.genre)

        X_augmented = pd.DataFrame(
            {'message': new_text, 'genre': genre}, index=idx)

        y_augmented = y.loc[idx]

        X_sub_new = pd.concat([X, X_augmented])
        y_sub_new = pd.concat([y, y_augmented])

        X_sub_new = X_sub_new.astype(X.dtypes.to_dict())
        y_sub_new = y_sub_new.astype(y.dtypes.to_dict())

        return X_sub_new, y_sub_new, self.tail_labels

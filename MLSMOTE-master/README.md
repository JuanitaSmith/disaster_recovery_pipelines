# MLSMOTE
Multi label Synthetic Minority Over-sampling Technique (MLSMOTE)

## Introduction
MLSMOTE is a technique used to handle data imbalance in multi-label classification problems. This implementation provides an updated version of the original MLSMOTE algorithm, with additional features and improvements.

## Changes
* **Updated to work correctly with pandas DataFrames as input**: Converted to numpy arrays before resampling and back to DataFrames after resampling.
* **Added `input_columns` and `label_columns` arguments**: Specify the input columns and labels for resampling.

## Resources
For more information on MLSMOTE and its applications, please refer to:
* [Handling Data Imbalance in Multi-Label Classification: MLSMOTE](https://medium.com/thecyphy/handling-data-imbalance-in-multi-label-classification-mlsmote-531155416b87)
* [Imbalanced-learn: MLSMOTE implementation](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/70e1f966414cdf9425d764c52a0a92d99c732051/imblearn/over_sampling/_mlsmote.py)

## Usage
Provided in `example.py` file.

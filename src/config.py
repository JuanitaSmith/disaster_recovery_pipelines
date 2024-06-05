# SAGEMAKER SET-UP

# AWS role when working locally
# Udacity
# sagemaker_role = 'arn:aws:iam::575658184623:role/service-role/AmazonSageMaker-ExecutionRole-20210428T162000'

# Own account
sagemaker_role = 'arn:aws:iam::588541864162:role/SageMakerAutoAccess'

# default s3 bucket name
prefix = 'disaster_pipeline'

# path to directories where raw, clean, config data and model outputs are stored
data_dir = '../data'
path_raw = '../data/raw'
path_clean = '../data/clean'
path_train = '../data/train/train'
path_validation = '../data/train/validation'
path_validation_test = '../data/train/test'
path_test = '../data/test'
path_output = '../data/output'
path_output_validation = '../data/output/validation'
path_output_test = '../data/output/test'
path_output_kaggle = '../data/output/kaggle'
path_model = '../data/models'
path_config = '../data/config'

# points to prepared training and testing files
filename_train_csv = 'train.csv'
filename_validation_csv = 'validation.csv'
filename_test_csv = 'test.csv'
filename_train_parquet = 'train.parquet'
filename_validation_parquet = 'validation.parquet'
filename_test_parquet = 'test.parquet'
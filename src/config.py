import os
# SAGEMAKER SET-UP

# AWS role when working locally
# Udacity
# sagemaker_role = 'arn:aws:iam::575658184623:role/service-role/AmazonSageMaker-ExecutionRole-20210428T162000'

# Own account
# sagemaker_role = 'arn:aws:iam::588541864162:role/SageMakerAutoAccess'
#
# # default s3 bucket name
# prefix = 'disaster_pipeline'
#
# # path to directories where raw, clean, config data and model outputs are stored
# data_dir = '../data'
# path_raw = '../data/raw'
# path_clean = '../data/clean'
# path_train = '../data/train/train'
# path_validation = '../data/train/validation'
# path_validation_test = '../data/train/test'
# path_test = '../data/test'
# path_output = '../data/output'
# path_output_validation = '../data/output/validation'
# path_output_test = '../data/output/test'
# path_output_kaggle = '../data/output/kaggle'
# path_model = '../data/models'
# path_config = '../data/config'
#
# # points to prepared training and testing files
# filename_train_csv = 'train.csv'
# filename_validation_csv = 'validation.csv'
# filename_test_csv = 'test.csv'
# filename_train_parquet = 'train.parquet'
# filename_validation_parquet = 'validation.parquet'
# filename_test_parquet = 'test.parquet'

# project_root = os.getcwd()
project_root = os.path.dirname(os.path.abspath(__file__))
print('Project root: {}'.format(project_root))

# FOLDER NAMES
folder_data = 'data'
folder_raw = 'data/raw'
folder_clean = 'data/clean'
folder_embeddings = 'data/embeddings'
folder_logs = 'logs'
folder_db = 'sqlite:///'
folder_scripts = 'src'
folder_models = 'models'
folder_translations = 'data/translations'

# LOG NAMES
filename_log_process_data = 'process_data.log'
filename_log_train_classifier = 'train_classifier.log'
filename_log_translation = 'translation.log'

# FILE/TABLE NAMES
filename_messages = 'disaster_messages.csv'
filename_categories = 'disaster_categories.csv'
filename_database = 'DisasterResponse.db'
filename_model = 'model_xgb_disaster_pipeline.pkl'
filename_translations = 'translations.csv'
filename_translations_log = 'translations.log'
filename_translations_json_batchjob = 'batch_tasks_language_detection.jsonl'
filename_translations_json_batchjob_results = 'batch_job_results.jsonl'

# FILE PATHS
path_log_process_data = os.path.join(project_root, folder_logs, filename_log_process_data)
path_log_train_classifier = os.path.join(project_root, folder_logs, filename_log_train_classifier)
path_log_translation = os.path.join(project_root, folder_logs, filename_translations_log)
path_messages = os.path.join(project_root, folder_raw, filename_messages)
path_categories = os.path.join(project_root, folder_raw, filename_categories)
path_database = os.path.join(folder_db, '..', folder_clean, filename_database)
path_model = os.path.join(project_root, folder_models, filename_model)
path_translations = os.path.join(project_root, folder_translations, filename_translations)
path_translation_json_batchjob = os.path.join(project_root, folder_translations, filename_translations_json_batchjob)
path_translation_json_batchjob_result = os.path.join(project_root,
                                                     folder_translations,
                                                     filename_translations_json_batchjob_results)

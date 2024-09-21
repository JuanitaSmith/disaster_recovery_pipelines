""" default path and file locations """
import os

# get working directory
project_root = os.getcwd()

# strip file path to the root directory of the project
project_root = project_root.split(sep='/src')[0]
project_root = project_root.split(sep='/tests')[0]
project_root = project_root.split(sep='/notebooks')[0]
project_root = project_root.split(sep='/models')[0]
project_root = project_root.split(sep='/app')[0]

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
filename_log_ml_prep = 'train_classifier_prep.log'
filename_log_ml = 'train_classifier.log'
filename_log_translation = 'translation.log'

# FILE/TABLE NAMES
filename_messages = 'disaster_messages.csv'
filename_categories = 'disaster_categories.csv'
filename_database = 'DisasterResponse.db'
filename_database_optuna = 'DisasterResponseStudyOptuna.db'
filename_model_prep = 'classifier_prep.pkl'
filename_model = 'classifier.pkl'
filename_optuna_study = 'optuna_study.pkl'
filename_translations = 'translations.csv'
filename_translations_json_batchjob = 'batch_tasks_language_detection.jsonl'
filename_translations_json_batchjob_results = 'batch_job_results.jsonl'
filename_model_featuremap = '../data/models/xgb.fmap'

# LINKS TO TABLE NAMES IN SQlite DB
table_messages = 'messages'
table_converted_messages = 'message_language'

# FILE PATHS
path_log_process_data = os.path.join(project_root, folder_logs, filename_log_process_data)
path_log_ml_prep = os.path.join(project_root, folder_logs, filename_log_ml_prep)
path_log_ml = os.path.join(project_root, folder_logs, filename_log_ml)
path_log_translation = os.path.join(project_root, folder_logs, filename_log_translation)
path_messages = os.path.join(project_root, folder_raw, filename_messages)
path_categories = os.path.join(project_root, folder_raw, filename_categories)
path_database = (folder_db + project_root + '/' + folder_clean + '/' + filename_database)
path_database_optuna = (folder_db + project_root + '/' + folder_clean + '/' + filename_database_optuna)
path_model_prep = os.path.join(project_root, folder_models, filename_model_prep)
path_model = os.path.join(project_root, folder_models, filename_model)
path_model_optuna_study = os.path.join(project_root, folder_models, filename_optuna_study)
path_translations = os.path.join(project_root, folder_translations, filename_translations)
path_translation_json_batchjob = os.path.join(project_root, folder_translations, filename_translations_json_batchjob)
path_translation_json_batchjob_result = os.path.join(project_root,
                                                     folder_translations,
                                                     filename_translations_json_batchjob_results)

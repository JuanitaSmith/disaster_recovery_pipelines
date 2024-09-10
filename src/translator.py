""" Class to convert non_english text to english """

import json
import logging
import sys
import time

import numpy as np
import pandas as pd
from openai import OpenAI
from sqlalchemy import create_engine

from src import config

# activate logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename=config.path_log_translation,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    filemode='w',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


class OpenAITranslator:
    """ Class to detect words that not in english and translate them

    Experiment with CHATGPT to detect the language of text messages and translate it to english

    Note: CHATGPT is a paid service, and takes a long time to translate.
    Just for general experimentation, only a sample of the data will be converted and using batch mode.
    Translations will be stored in json format in ../data/translations/batch_job_results
    Update messages database with converted/improved message.

    This class was designed to convert only a portion of the messages for experimental reasons

    """

    def __init__(self, openai_api_key, translation=True, max_text_to_translate=2000):

        print('Initialization....')

        # OPENAI request
        self.system_prompt = """ 
          You will be provided with text about disaster responses.
          Step 1: Detect if the text is in English. Return options 'True' if the sentence is in English or 
          'False' if the sentence is not in English as isEnglish boolean variable
          Step 2: If sentence is not in English, translate it to English and return as text in json format

          Example: 'I need food' // isEnglish: True
          Example2: 'Vandaag is het zonnig'' // isEnglish: False, Translation: 'Today it is Sunny'
          """

        self.end = int(max_text_to_translate)
        self.start = 0
        self.batch_size = 400
        # self.model = 'gpt-4-turbo'
        self.model = 'gpt-4o'
        self.temperature = 0.1

        if translation in ['True', 'true', True]:
            self.translate = True
        elif translation in ['False', 'false', False]:
            self.translate = False
        else:
            print('translation does not have values True or False - {}'.format(translation))
            logger.error('translation does not have values True or False - {}'.format(translation))
            sys.exit(1)

        # setup connection to OPENAI if translation is needed
        if translation:
            self.openai_api_key = openai_api_key
            try:
                # self.openai_api_key = os.environ.get('OPENAI_API_KEY')
                self.client = OpenAI(api_key=self.openai_api_key)
                # simple test to see if authentication is successful
                response = (self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_prompt,
                            "temperature": self.temperature,
                            "response_format": {
                                "type": "json_object"
                            },
                        },
                        {
                            "role": "user",
                            "content": 'Vandaag is het zonnig'
                        }
                    ],
                    temperature=self.temperature,
                ))
            except Exception as error:
                print('Missing or incorrect OPENAI API key - {}'.format(error))
                logger.error('Missing or incorrect OPENAI API key - {}'.format(error))
                sys.exit(1)

        logger.info('OPENAI API connection established')
        print('OPENAI API connection established')

        # setup connection to SQLITE database
        self.table_messages = 'messages'
        self.table_converted_messages = 'message_language'
        try:
            self.engine = create_engine(config.path_database)
            self.conn = self.engine.connect()
        except Exception as error:
            print('Missing or incorrect DATABASE at {} ({})'.format(config.path_database, error))
            logger.error('Missing or incorrect DATABASE at {} ({})'.format(config.path_database, error))
            sys.exit(1)

        print('Database connection established ({}_'.format(config.path_database))
        logger.info('Database connection established ({}_'.format(config.path_database))

    def load_data(self):
        """ Load data to be translated here

        For my scenario, I load data from SQLITE database
        Two datasets are required:

        Args:
            filepath -> string: Link to SQLITE database

         Returns:
            df -> pd.DataFrame: data containing text to translate
            df_translations -> pd.DataFrame: data containing messages already translated

         """

        print('Loading data...')
        logger.info('Loading data...')

        # load text messages from database created during ETL pipeline preparation
        sql_statement = 'select * from {}'.format(self.table_messages)
        df_messages = pd.read_sql(sql=sql_statement, con=self.conn, index_col='id')

        # try to read message where language detection was already executed, if it exists
        try:
            sql_statement = 'select * from {}'.format(self.table_converted_messages)
            df_translations = pd.read_sql(sql=sql_statement, con=self.conn, index_col='id',
                                          dtype={'is_english': 'boolean'})
        except Exception as info:
            # df_translations = pd.DataFrame()
            logger.error('No existing translations were found in database, even though it already exists')
            sys.exit(1)

        print('Number of messages found {}'.format(df_messages.shape[0]))
        print('Number of messages already converted: {}'.format(df_translations.shape[0]))
        logger.info('Number of messages found {}'.format(df_messages.shape[0]))
        logger.info('Number of messages already converted: {}'.format(df_translations.shape[0]))

        return df_messages, df_translations

    def _translate(self, df_messages, df_translations, json_batch_file=config.path_translation_json_batchjob):
        """ Create OPENAI batches to translate the next_iter sample of text messages

        Args:
            df_messages -> pd.DataFrame: text messages to check and translate
            df_language -> pd.DataFrame: text messages already processed and checked with OPENAI
            json_batch_file -> str: file path where json batch data will be stored that are uploaded to OPENAI

        Returns:
            batches -> list: list of batch numbers created and processed successfully on OPENAI
        """

        print('Translation started...')
        logger.info('Translation started...')

        # Identify messages that was not yet selected for checking and translation
        if df_translations.shape[0] > 0:
            df_remaining = df_messages.merge(df_translations, on='id', how='left', indicator=True)
            df_remaining = df_remaining[df_remaining['_merge'] == "left_only"]
        else:
            df_remaining = df_messages

        # Creating an array of json tasks for the next_iter 2000 texts messages in batches containing 400 messages each
        next_iter = self.start + self.batch_size
        batches = []

        while next_iter <= self.end:

            # create an array of json tasks for each batch job
            tasks = []
            for index, text in df_remaining['message'][self.start:next_iter].items():
                task = {
                    "custom_id": f"task-{index}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        # This is what you would have in your Chat Completions API call
                        "model": self.model,
                        "temperature": self.temperature,
                        "response_format": {
                            "type": "json_object"
                        },
                        "messages": [
                            {
                                "role": "system",
                                "content": self.system_prompt,
                            },
                            {
                                "role": "user",
                                "content": text,
                            },
                        ]
                    }
                }

                tasks.append(task)

            # create json file and save it locally
            with open(json_batch_file, 'w') as file:
                for obj in tasks:
                    file.write(json.dumps(obj) + '\n')

            # Uploading json file to openai platform
            batch_file = self.client.files.create(
                file=open(config.path_translation_json_batchjob, 'rb'),
                purpose='batch'
            )

            # Creating the batch job on openai
            batch_job = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )

            print('Batch submitted {} for records {}-{}'.format(batch_job.id, self.start, next_iter))
            logger.info('Batch submitted {} for records {}-{}'.format(batch_job.id, self.start, next_iter))

            # Check status of batch job running on openai platform
            print('Waiting for batch to start, go to sleep 5 minutes')
            time.sleep(300)
            batch_job = self.client.batches.retrieve(batch_job.id)
            print('Batch {}" status {}'.format(batch_job.id, batch_job.status))
            logger.info('Batch {}" status {}'.format(batch_job.id, batch_job.status))

            # wait for batch to complete before starting the next_iter batch job.
            # NOTE: CHATGPT does not allow multiple batches to run in parallel
            while batch_job.status in ['in_progress', 'validating', 'finalizing']:
                print('Batch {} still running - going to sleep for 5 minutes'.format(batch_job.id))
                logger.info('Batch {} still running - going to sleep for 5 minutes'.format(batch_job.id))
                time.sleep(300)
                batch_job = self.client.batches.retrieve(batch_job.id)

            # when batch is completed, set counters to kick off the next_iter batch job of 400 requests
            if not batch_job.status == 'failed':
                self.start += self.batch_size
                next_iter += self.batch_size
                batches.append(batch_job)

            logger.info('Batches successfully processed on OPENAI'.format(batches))

        return batches

    def _batch_translations_to_json(self, json_file=config.path_translation_json_batchjob_result):
        """ Download batch jobs results that were submitted on openai and accumulate their results in a local json file

        Args:
            json_file -> str: file path to accumulate and store all json results from all batches
        """

        print('Extracting batches into local json files...')
        logger.info('Extracting batches into local json files...')

        # get all batch jobs that were submitted on openai
        batch_jobs = self.client.batches.list(limit=100)
        print('Number of batch jobs retrieved: {}'.format(len(batch_jobs.data)))

        # select batches we want to save the results of
        # only select batches that completed, no records failed and the request count > 10, this eliminates trials
        batches = []
        for batch in batch_jobs.data:
            if ((batch.status == 'completed') &
                    (batch.request_counts.failed == 0) &
                    (batch.request_counts.total > 10)):
                batches.append(batch.id)
                print(batch.id, batch.status, batch.request_counts.completed, batch.request_counts.failed)
                logger.info('Batch id {}, Status: {}, Completed: {}, Failed: {}'.format(
                    batch.id, batch.status, batch.request_counts.completed, batch.request_counts.failed))

        # Download batch content from OPENAI and consolidate all api results locally into a json file
        # OPENAI will delete files from batches after 30 days, so we might not be able to retrieve old content
        # file batch_job_results will still contain all those old details so don't delete it, just append

        # append contents of all batches to local json results file
        for batch in batches:
            batch_job = self.client.batches.retrieve(batch)
            try:
                result = self.client.files.content(batch_job.output_file_id).content
                # append contents, keep details of old results, we will remove duplicates later
                with open(json_file, 'ab') as file:
                    file.write(result)
                logger.info('Batch file {} output successfully retrieved'.format(batch))
            except Exception as info:
                # print('Batch file {} output already deleted on openai'.format(batch))
                logger.info('Batch file {} output already deleted on openai - {}'.format(batch, info))

    def _json_translations_to_pandas(self, json_file=config.path_translation_json_batchjob_result):
        """ Load all json api data from locally saved json file into a pandas dataframe

        Args:
            df_messages -> pd.DataFrame: pandas dataframe containing original messages
            json_file -> str: file path where all batch results were stored in json format

        Return:
            df_translations -> pd.DataFrame: dataframe containing all text translations where it was needed

        """

        print('Save translations into pandas dataframe ...')
        logger.info('Save translations into pandas dataframe ...')

        # Loading all json api results from locally saved json file
        results = []
        with open(json_file, 'r') as file:
            for line in file:
                # Parsing the JSON string into a dict and appending to the list of results
                json_object = json.loads(line.replace('\n', '').strip())
                results.append(json_object)

        # Load all responses into a dictionaries
        isEnglishs = {}
        translatedTexts = {}

        for res in results:
            task_id = res['custom_id']
            # Get unique message id from task id
            index = int(task_id.split('-')[-1])
            # get response content and strip of new line indicators
            result = res['response']['body']['choices'][0]['message']['content']
            result = result.replace('\n', '').strip()
            result = result.replace('\t', '').strip()
            # # get original message
            # df_tmp = df_messages.loc[index]
            # description = df_tmp['message']
            translation = ''
            isEnglish = ''

            try:
                dict_object = json.loads(result)
                isEnglish = dict_object['isEnglish']
            except Exception as error:
                logger.info('index {} - could not be parsed ({}) - {}'.format(index, error, dict_object))

            try:
                translation = dict_object['Translation']
            except Exception as info:
                # if a message is already in english, the translation will not be supplied, it's not really an error
                # logger.info('index {} - could not be parsed ({}) - {}'.format(index, info, dict_object))
                pass

            isEnglishs[index] = isEnglish
            translatedTexts[index] = translation

        # create dataframe contain translator results
        data = {'is_english': isEnglishs,
                'translation': translatedTexts}
        df_translations = pd.DataFrame.from_dict(data, orient='columns')
        df_translations.index.name = 'id'

        print('Number of translated messages found in json files before cleaning: {}'.format(df_translations.shape[0]))
        logger.info('Number of translated messages found in json files before cleaning: {}'.format(
            df_translations.shape[0]))
        return df_translations

    def _clean_translations(self, df):

        print('Cleaning translations...')
        logger.info('Cleaning translations...')

        # strip spaces from 'is_english' and capitalize
        df['is_english'] = df['is_english'].map(lambda x: str(x).replace(' ', '').capitalize())

        # convert to boolean
        df['is_english'] = df['is_english'].map(lambda x: True if str(x) == 'True' else False)

        # if text is not marked as English, but no translation was given, set the language back to English
        error_index = df[(~df['is_english']) & (df['translation'].map(lambda x: len(x) == 0))].index
        logger.info('Non-English text without translations found and corrected: {}'.format(len(error_index)))
        df.loc[error_index, 'is_english'] = True

        # drop duplicated translations if it exists
        df = df[~df.index.duplicated(keep='first')]

        print('{} messages were checked and translated with OpenAI after cleaning'.format(df.shape[0]))
        print('Messages not in English after cleaning: {}'.format(len(
            df[~df.is_english])))

        logger.info('{} messages were checked and translated with OpenAI after cleaning'.format(
            df.shape[0]))
        logger.info('Messages not in English after cleaning: {}'.format(
            len(df[~df.is_english])))

        df = df.sort_index()
        return df

    def _save_translations(self, df):
        """ save translations to SQLite database """

        print('Saving translations...')
        logger.info('Saving translations...')

        # add to existing sqlite database
        df.to_sql(self.table_converted_messages, self.engine, index=True, if_exists='replace')

    def _update_messages(self, df_messages, df_translations):
        """ Update messages dataset with translations and update SQLite database with an improved messages dataset"""

        print('Updating messages dataset with translations...')
        logger.info('Updating messages dataset with translations...')

        # merge dataframes
        if len(df_translations) > 0:
            df_messages = df_messages.merge(df_translations, on='id', how='left')
            # copy the original message
            df_messages['original_message'] = df_messages['message']

            # For messages that were not yet translated, set is_english = True
            df_messages['is_english'] = np.where(df_messages['is_english'].isna(), True, df_messages['is_english'])

            # replace message with translation, if message is flagged as not being in English
            df_messages['message'] = np.where(
                (df_messages['is_english'] == False) & (~df_messages['translation'].isnull()),
                df_messages['translation'],
                df_messages['original_message'])

            # finally lets drop the columns we no longer need
            df_messages = df_messages.drop(['original_message', 'is_english', 'translation'], axis=1)

            # save updated messages back to database in exactly the same format, to make language translation optional
            df_messages.to_sql('messages', self.engine, index=True, if_exists='replace')

            print('Messages successfully updated with translated texts')
            logger.info('Messages successfully updated with translated texts')
        else:
            print('No translations found, messages are not updated')
            logger.info('No translations found, messages are not updated')

    def main(self):
        """ Main routine

        If `self.translate` is False, translation results already stored in json result file,
        will be accumulated and saved to SQLite database

        If `self.translate` is True, translation steps will be triggered first
        """

        print('Main routine triggered....')

        df_messages, df_translations = self.load_data()

        if self.translate:
            self._translate(df_messages, df_translations)
            self._batch_translations_to_json()

        df_translations = self._json_translations_to_pandas()
        df_translations = self._clean_translations(df_translations)
        self._save_translations(df_translations)
        self._update_messages(df_messages, df_translations)

        print('Translation completed !!!')
        logger.info('Translation completed !!!')


if __name__ == '__main__':

    if len(sys.argv) == 4:
        api_key, translate, max_records_to_translate = sys.argv[1:]

        translate = translate.capitalize()
        if translate not in ['True', 'False']:
            print('\nPlease enter values True or False for the second parameter related to translation')
            sys.exit(1)

        trans = OpenAITranslator(openai_api_key=api_key,
                                 translation=translate,
                                 max_text_to_translate=max_records_to_translate)
        trans.main()

    else:
        print('\nPlease provide the api key for OpenAI, Boolean indicator (True or False) to trigger translation',
              'and the maximum number of messages to translate '
              'as the first, second and third argument respectively. \n\nExample: python translater.py '
              '■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■gTG2jqP8, True, 400.'
              '\nThis will check and translate the next unchecked 400 messages if it is not in English.'
              '\n\nExample: Text message for index 9874 will be translated from'
              '\n`2O TRI PYE, 5 POIRO, 2GD PESI, 1OMORU, 5LAY3DOLA SITWON` to '
              '\n`20 bags of rice, 5 bottles of water, 2 packs of peas, 1 tomato, 5 layers of dollars situation`')

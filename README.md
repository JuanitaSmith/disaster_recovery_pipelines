# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


TODO:
conda install -c conda-forge --yes --file requirements.txt
conda install -c jmcmurray json



AWS setup

Setup

To get started, complete the following steps:

Create a new user with programmatic access that enables an access key ID and secret access key for the AWS CLI.
Attach the permissions AmazonSageMakerFullAccess and AmazonS3FullAccess.
Limit the permissions to specific Amazon Simple Storage Service (Amazon S3) buckets if possible.
You also need an execution role for the SageMaker AmazonSageMakerFullAccess and AmazonS3FullAccess permissions. SageMaker uses this role to perform operations on your behalf on the AWS hardware that is managed by SageMaker.
Install the AWS CLI on your local computer and quick configuration with aws configure:
$ aws configure
AWS Access Key ID [None]: AKIAI*********EXAMPLE
AWS Secret Access Key [None]: wJal********EXAMPLEKEY
Default region name [None]: eu-west-1
Default output format [None]: json
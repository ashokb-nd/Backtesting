import boto3
import configparser
import os
import logging

# Set up logging to a file as well as the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', 
                    handlers=[logging.FileHandler('./fetch_all_models.log'), logging.StreamHandler()],
                    datefmt='%Y-%m-%d %H:%M:%S')

# Load the configuration file
config = configparser.ConfigParser()
# config.read('/data4/kaushik/reprocess_dir/analytics/src/nd_config_bagheera2_US.ini')
config.read('/data4/ashok/analytics/src/nd_config_bagheera2_US.ini')
# Specify the local path where you want to download the models
local_path = '/data4/ashok/analytics/cache/autocam'
# local_path = '/data/nd_files/autocam'

# Get the model list and base aws path
base_aws_path = 'analytics/models/'
model_list = [ os.path.basename(path) for key, path in config.items('deviceModelFiles') if key.endswith('_path')]
logging.info(f'Found {len(model_list)} models in the configuration file.')
# Initialize the S3 client
s3 = boto3.client('s3')

# Set force_download to True if you want to download all models regardless of whether they exist locally
force_download = False

# Loop through the model list
for model in model_list:
    # Construct the local model path
    local_model_path = os.path.join(local_path, model)

    # Check if the model exists locally
    if os.path.exists(local_model_path) and not force_download:
        logging.info(f'Model {model} already exists locally. Skipping download.')
        continue

    # Construct the S3 path
    s3_path = os.path.join(base_aws_path, model)

    # Check if the model exists in the S3 bucket
    # try:
    #     s3.head_object(Bucket='netradyne-sharing', Key=s3_path)
    #     logging.info(f'Model {model} exists in the S3 bucket.')
    # except Exception as e:
        
    #     logging.error(f'Model {model} does not exist in the S3 bucket.:{e}')
    #     continue

    # Download the model from S3 to the desired location
    try:
        print('aws s3 cp s3://{}/{} {} --recursive'.format('netradyne-sharing',s3_path,local_model_path))
        os.system('aws s3 cp s3://{}/{} {} --recursive'.format('netradyne-sharing',s3_path,local_model_path))
        # s3.download_file('netradyne-sharing', s3_path, local_model_path)
        if os.path.exists(local_model_path):
            logging.info(f'Model {model} has been downloaded.')
    except Exception as e:
        logging.error(f'Failed to download model {model}. Error: {str(e)}')

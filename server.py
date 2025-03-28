from flask import Flask, render_template, request
import os
import time
import joblib
import numpy as np
import boto3
import csv
from datetime import datetime
import pandas as pd
from waitress import serve
import logging


logging.basicConfig(
    filename='transaction_log.log',  # Log file name
    level=logging.INFO,                 # Log level (INFO, DEBUG, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

#load the model
model = joblib.load('fraud_model.pkl')

field_names = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order"
] 

LOG_FILE = 'transaction_log.csv'
TEMP = 'temp.csv'
# Create a CSV file with the field names if it doesn't exist
try: 
    with open(LOG_FILE, mode='x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['times_stamp'] + field_names + ['prediction', 'verification'])
        logging.info('Application has started ...')
except FileExistsError:
    pass # File already exists
    logging.info('transaction_log file already exists...')

try: 
    with open(TEMP, mode='x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['times_stamp'] + field_names + ['prediction', 'verification'])
        logging.info('Temp file created...')
except FileExistsError:
    pass # File already exists

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    print("Request files:", request.files)  # Examine the request files

    # Error handling for file upload  
    if 'file' not in request.files:
        logging.error("No file in the request")
        print("No file in the request")
        return render_template("error.html", error_message = "Error: No file in the request")
    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected")
        return render_template("error.html", error_message = "Error: No file selected")
    if not file.filename.endswith('.csv'):
        logging.error("Invalid file type")
        return render_template("error.html", error_message = "Error: Please upload a CSV file" )
    try:
        df = pd.read_csv(file)
        logging.info('Successfully uploaded file...')
        print(df.head())

        # Check if the required columns are present
        missing_columns = [col for col in field_names if col not in df.columns]
        if len(missing_columns) > 0:
            return render_template("error.html", error_message = "Error: Missing columns in the CSV file: \n" + ", ".join(missing_columns))
        
        # predict fraud
        def process(data):
            predictions = model.predict(data.values)
            data['prediction'] = ['Fraudulent' if pred == 1 else 'Legitimate' for pred in predictions]
            data.insert(0, 'timestamp',  datetime.now())
            return data
        
        data = process(df)    

        # Add verification column
        data['verification'] = np.where(data['prediction'] == 'Legitimate', 'yes', '')


        # Save the data to a CSV file
        split_data = pd.concat([data[data['prediction'] == 'Legitimate'], data[data['prediction'] == 'Fraudulent']], ignore_index=True, sort=False)
        print(split_data.head(24))
        split_data.to_csv(TEMP, mode='w', header=True, index=False)  

        #save the data to the log file
        if split_data[split_data['verification'] != 'yes'].shape[0] == 0:
            split_data.to_csv(LOG_FILE, mode='a', header=False, index=False)
        else:
            pass


        # Save the data to a dictiory for rendering in the template
        fraud_transactions = data[data['prediction'] == 'Fraudulent'].to_dict(orient='records')
        logging.info('file successfully processed ...')

        return render_template("verify.html", title="Upload Successful", transactions=fraud_transactions)
    except Exception as e:
        return f"Error processing file: {str(e)}", 500

@app.route('/verify', methods=['POST'])
def verify_transactions():
    verified_data = request.form.to_dict()
    logging.info('Response dictionrary successfully sent to veryify transactions...')
    data = pd.read_csv(TEMP)

    # View data from request
    print(f"Verified data: {verified_data}")
    
    # Adding Verification to the log file
    fraud_count = sum(1 for key in verified_data.keys() if key.startswith('verify_'))
    fraud_data = data.tail(fraud_count)
    verify_list = [verified_data[f'verify_{idx}'] for idx in range(fraud_count)]
    fraud_data['verification'] = verify_list
    old_data = data.head(len(data) - fraud_count)
    total_data = pd.concat([old_data, fraud_data], ignore_index=True, sort=False)
    total_data.to_csv(LOG_FILE, mode='a', header=False, index=False)
    logging.info('All fradulent transactions verified...')
    print(total_data.head(24))

    # Check log file size
    file_size = pd.read_csv(LOG_FILE).shape[0]
    if file_size > 1000:
        for _ in range(25):
            print('*******'*10)
            print("Log file has a full batch. Model would need to be retrained")

    logging.info('file successfully stored in DB... ')

    # Upload to S3 bucket
    #s3 = boto3.client('s3')
    #bucket_name = 'fd-sagemaker-bucket'
    #s3.upload_file(LOG_FILE, bucket_name, LOG_FILE)
    #print(f"Log file uploaded to S3 bucket {bucket_name}")
    
    # Log file is also saved locally to cloud storage 

    return render_template("success.html")

if __name__ == '__main__':
    logging.info("Starting Waitress server...")
    serve(app, host='0.0.0.0', port=8080)
    logging.info("Waitress server started.")
# Fraud Detection Application Setup and Usage Guide

Fraud Detection Application Cloud Portal: https://capstone-fraud-detection.onrender.com 

The application was designed with batch processing in mind. However on a slow day when one transaction needs to be verified the app can take this form: https://analyse-transaction.onrender.com/

This document provides step-by-step instructions to install, build, and run a Flask-based web application designed to upload CSV transaction files, detect fraudulent transactions, and allow human verification. The application consists of a home page for file uploads (index.html), a verification page for flagged transactions (verify.html), an error page for handling issues (error.html), and a consistent styling file (styles.css). 

All data whill be logged and stored for retraining. The verification is aspect is critically important as it allows drift monitoring primairly data drift and concept drift. 

### Prequisites

##### Before setting up the project, ensure you have the following installed on your system
1. Python 3.8+
    - Download from python.org.
    - Verify installation: python --version or python3 --version.
2. pip
    - Python’s package manager, usually included with Python.
    - Verify: pip --version or pip3 --version.
3. A Text Editor
    - Recommended: Visual Studio Code, PyCharm, or any editor of your choice.
4. Web Browser
    - Any modern browser (e.g., Chrome, Firefox) to access the application.

### Application Structure
##### After setup, your project directory should look like this:
<pre>
Fraud-detection-app
├── static/
│   └── styles/
│       └── styles.css
├── templates/
│   ├── index.html
│   ├── verify.html
│   └── error.html
├── server.py
├── fraud_model.pkl
├── requirements.txt
├── test.csv (5 files)
└── README.md
</pre> 
- static/styles/styles.css: CSS file for styling all pages.
- templates/: Directory containing HTML templates.
- server.py: Main Flask application script.
- requirements.txt: List of Python dependencies.

### Installation and Setup 
##### Step 1: Create Project Directory (e.g: fraud-detection-app) 
1. Dowloand all folders and files
2. Create a directory system as above in your created folder
3. Open your text Editor and Navigate to your folder. 
4. Ensure the directory is set up as above

##### Step 2: Set Up a Virtual Environment
1. Open you terminal
2. Create a virtual environment:
    -  Windows: python -m venv venv
3. Activate it:
    - Windows: venv\Scripts\activate
    - macOS/Linux: source venv/bin/activate
    - You should see (venv) in your prompt.

##### Step 3: Install Dependencies
- Assuming a directory system as above
- install dependencies: run the following in the terminal *pip install -r requirments.txt*

##### Step 4: Your program is ready to run. 

### Building and Running the Application
##### Step 1: Verify Files
Ensure all files (server.py, index.html, verify.html, error.html, styles.css) are in the correct directories as shown above.

##### Step 2: Run the Application
1. Run the Flask app: 'python server.py'
2. You should see output like: 'Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)' 

##### Step 3: Access the Application
1. Open a web browser.
2. Navigate to: http://127.0.0.1:5000/
3. You’ll see the "Upload Transaction Document" page.

### Using the Application
##### Step 1: Upload a CSV File
On the home page (http://127.0.0.1:5000/):
    - Click "Choose File" and select a CSV file with the following columns:  [distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price, repeat_retailer, used_chip, used_pin_number, online_order] 
    - You can upload a test file
    - Click Upload

##### Step 2: Verfy Fradulent Transations 
If the CSV is valid and contains fraudulent transactions (fraud == 1):
  - You’ll be redirected to the verification page (/verify).
  - A table displays flagged transactions 
  - For each row, select "Yes" or "No" to verify fraud.
  - Click "Submit Verifications" 
If no fraudulent transactions are detected:
  - You’ll see a message: "No fraudulent transactions detected."

##### Step 3: Handle Errors
If there’s an issue (e.g., no file, invalid CSV):
- You’ll see the error page with a message (e.g., "CSV missing required columns", "No file selected" etc.).
- Click "Return to Home" to try again.

##### Retraining: You will be prompted to retrain the model after a certain number of entries have been logged. 

### Troubleshooting
- 404 Errors: Ensure all routes (/, /upload) are defined in server.py.
- CSS Not Loading: Check the static/styles/styles.css path and Flask’s url_for.
- Table Misalignment: Verify table-layout: fixed in styles.css.


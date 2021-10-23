# Disaster response pipeline

### **Instructions**:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### **Project Motivation**:

This project (Disaster Response Pipelines) is part of Udacity's Data Scientists Nanodegree Program.

This project focuses on the following aspects: 

1. Implement diverse ETL steps, such as merging datasets and cretaing dummy columns for categorical 
   features
2. Set up a Machine Learning Pipeline containing a Count Vectorizer and Tfidf Transformer, since the
   X data consists of messages = text data
3. Implement a web app where a message can be passed into an input field on the frontend, and the
   classification into the 36 categories is also displayed on the frontend.
  
### **Libraries**:
Python3

Here are the libraries I used in my Jupyter Notebook:

numpy

pandas

sklearn

flask

plotly


### **File Descriptions**:
process_data.py: implements the ETL steps

train_classifier.py: implements the ML learning pipeline including vectorization
run.py: sets up the web app with flask in the backend


### **Summary of the results**:
The result is a web app able to classify an inputted message into the 36 available
categories and output the result on the frontend.

![](./webappview.jpg)

### **Licensing, Authors, Acknowledgements**:
Must give credit to Figure Eight for the data and to Udacity for the idea and template of the project.
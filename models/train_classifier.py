# import libraries
import sys
import numpy as np
import pandas as pd
import pickle
import sqlite3
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

def load_data(database_filepath):
    """
    Description: This function can be used to read in the data from DisasterResponse.db and
    generate a pandas DataFrame thereof
    Arguments:
        database_filepath: the path to the DisasterResponse.db
    Returns:
        df: DisasterResponse data as pandas DataFrame
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('InsertTableName', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(df.columns[4:])
    return X, y, category_names


def tokenize(text):
    """
    Description: This function normalizes, tokenizes and lemmatizes an
    inputted text
    Arguments:
        text: the text to be normalized, tokenized, lemmatized
    Returns:
        lemmed: list of words contained in the text, in a lemmatized format
    """
    # normalize text = set words in text to lowercase and remove punctation
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    # initialise lemmatizer
    lemmatizer = WordNetLemmatizer()
    # lemmatize every token in token that is not a stopword
    lemmed = [lemmatizer.lemmatize(t) for t in tokens if t not in stopwords.words('english')]
    return lemmed


def build_model():
    """
    Description: This function sets up a pipeline consisting of a Count Vectorizer,
    customized with the tokenize function, a tfidf transformer and a
    Random Forest Classifier embedded in a Multi Output Classifier
    Arguments:
        None
    Returns:
        pipeline= the built pipeline
    """
    # build a machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Description: This function serves to make a prediction for the Y outputs,
    based on the X_test data. For each of the 36 a classification report and accuracy
    score is printed.
    Arguments:
        model: the model to be evaluated
        X_test: X_test data in the form of a Pandas Seriies
        y_test: y_test data in the form of a Pandas DataFrame
        category_names: list of the 36 category names
    Returns:
        None
    """
    predictions = model.predict(X_test)
    y_test_array = np.array(Y_test)
    for num in range(0, 36):
        real_values = [item[num] for item in y_test_array]
        predicted_values = [item[num] for item in predictions]
        print(classification_report(real_values, predicted_values))
        print(accuracy_score(real_values, predicted_values))

def apply_grid_search(model, X, y):
    """
    Description: This function applies a grid search on the model entered as an argument
    Arguments:
        model: the model to which the grid search is applied
        X: X data in the form of a Pandas Series
        y: y data in the form of a Pandas DataFrame
        category_names: list of the 36 category names
    Returns:
        None
    """
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2, 4]

    }
    # create grid search object
    cv = GridSearchCV(model, parameters)
    # fit grid search classifier
    cv.fit(X, y)
    return cv


def save_model(model, model_filepath):
    """
    Description: This function saves the model into a pickle file
    Arguments:
        model: the model object
        model_filepath: model filepath as entered into the command line: 'models/classifier.pkl'
    Returns:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Description: This function serves as a vehicle to run the other functions and
    indicate the steps in the process with print statements
    Arguments:
        None
    Returns:
        None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Applying grid search ...')
        cv = apply_grid_search(model, X, Y)
        print(cv.best_params_)
        print(cv.best_score_)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model=cv, model_filepath=model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
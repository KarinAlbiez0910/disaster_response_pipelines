import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Description: This function serves to generate the data for the graphs and creates the graphs
    themselves, which are handed over to the frontend
    Arguments:
        None
    Returns:
        render_template('master.html', ids=ids, graphJSON=graphJSON)
    """

    frequencies = {}
    for item in df.columns[4:]:
        try:
            value_1_percentage = df[item].value_counts().loc[1] / len(df)
            frequencies[item] = value_1_percentage
        except:
            print('No category 1 found')
    sorted_frequencies = {k: v for k, v in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}
    top_categories = [item[0] for item in list(sorted_frequencies.items()) if item[1] > 0.05]
    graphs = []
    for item in top_categories[1:13]:
        x = df[item].value_counts().index
        y = df[item].value_counts().values
        entry = {
            'data': [
                Bar(
                    x=x,
                    y=y
                )
            ],

            'layout': {
                'title': f'Distribution of Category {item}',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "0: messages not asigned to category, 1: messages assigned to category",
                    'tickvals': [0,1]
                }
            }
        }
        graphs.append(entry)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Description: This function contains the code to take the query made on
    the frontend and classify it with the machine learning model and output
    the result on the frontend
    Arguments:
        None
    Returns:
        render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Description: This function serves as a vehicle to run the other functions and
    indicate the steps in the process with print statements
    Arguments:
        None
    Returns:
        None
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
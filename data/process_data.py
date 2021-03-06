# import libraries

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Description: This function reads in the messages and categories csv datasets into Pandas DataFrames,
    merges them and generates dummy columns of the categories column, preparing the pd DataFrame for
    feeding it into a machine learning model
    Arguments:
        messages_filepath: data/disaster_messages.csv
        categories_filepath: data/disaster_categories.csv
    Returns:
        df: DisasterResponse data as a processed pandas DataFrame
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge messages and categories datasets
    df = messages.merge(categories, on='id')
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = list(categories.iloc[0])
    category_colnames = [item.split('-')[0] for item in category_colnames]
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert category values to just 1 or 0
    for column in categories:
        # set each value to be the last character of the string and convert it to numeric
        categories[column] = categories[column].apply(lambda x: int(x[-1]))
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    return df


def clean_data(df):
    """
    Description: This function cleans the dataset by removing duplicates from it
    Arguments:
        df: pandas DataFrame
    Returns:
        df: cleaned DataFrame with duplicates removed
    """
    # check number of duplicates
    np.sum(df.duplicated())
    # drop duplicates
    df = df.drop_duplicates()
    # check number of duplicates
    np.sum(df.duplicated())
    return df


def save_data(df, database_filename):
    """
    Description: This function saves the processed and cleaned DataFrame into a sqlite database
    Arguments:
        df: pandas DataFrame, processed (with category dummy columns) and cleaned
        database_filename: data/DisasterResponse.db
    Returns:
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('InsertTableName', engine, index=False)


def main():
    """
    Description: This function serves as a vehicle to run the other functions and
    indicate the steps in the process with print statements
    Arguments:
        None
    Returns:
        None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print(df.head())

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
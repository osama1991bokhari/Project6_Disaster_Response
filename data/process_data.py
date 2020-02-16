import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    - This function reads 2 csv files into a Pandas dataframe
    - Then returns a merged dataframe.
    Args:
    messages_file_path str: the messages csv file
    categories_file_path str: the categories csv file
    Returns:
    Merged_df pandas_dataframe: the merged dataframe of the two files
    """
    messages = pd.read_csv('messages.csv')
    categories = pd.read_csv('categories.csv')
    return messages.merge(categories, on='id')


def clean_data(df):
    """
    - This function cleans the merged dataframe and
    - prepares it for the machine learning part
    
    Args:
    df : the merged dataframe
    Returns:
    Cleaned data ready for the machine learning
    """
    # Acquiring the category names from the first row.
    categories = df['categories'].str.split(";",expand = True)
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    row_cat = row.str.split('-')
    category_colnames = []
    for column in row_cat:
        category_colnames.append(column[0])
        
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    this function will saves the cleaned data to an SQL database
    Args:
    df : the cleaned dataframe
    database_file_name : the db file where the cleaned dataframe
    will be stored into.
    data is to be saved
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_messages', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

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
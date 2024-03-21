import pandas as pd
import numpy as np

""" 
Data Loading and Preprocessing
"""

# Twitter data loading
twitter_path = 'python\data\\twitter_nov_2019.csv'
twitter_df = pd.read_csv(twitter_path)

### START FUNCTION
def word_splitter(df):
    """
    Split each tweet into individual words and store them in a new column.

    Args:
    df (DataFrame): DataFrame containing the 'Tweets' column.

    Returns:
    DataFrame: Modified DataFrame with a new column 'Split Tweets' containing lists of individual words.
    """
    pd.set_option('display.max_columns', None) # Set to display all columns in output

    # Extract the tweets as a list of strings
    twts_lst = list(df['Tweets'].values)

    # Split each tweet into individual words, convert to lowercase, and store them in a new column
    df["Split Tweets"] =  [[word.lower() for word in twt_str.split()] for twt_str in twts_lst]

    return df

### END FUNCTION

# Input:
word_splitter(twitter_df.copy())

# Output:
output_data = word_splitter(twitter_df.copy())
print(output_data)

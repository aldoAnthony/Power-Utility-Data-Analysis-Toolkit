import pandas as pd
import numpy as np


""" ### Data Loading and Preprocessing ### """

# Twitter data
twitter_path = 'python\data\\twitter_nov_2019.csv'
twitter_df = pd.read_csv(twitter_path)

### START FUNCTION
def word_splitter(df):
    pd.set_option('display.max_columns', None) # Set to display all columns in output

    twts_lst = list(df['Tweets'].values) # Converts 'Tweet 'column into a numpy array then into a list
    #print(twts_lst)# Observing output

    # Code takes a list of strings "twts_lst" and splits each string into a list of strings, storing all these new lists
    # into a column in the "df" dataframe named "Split Tweets"
    df["Split Tweets"] =  [[word.lower() for word in twt_str.split()] for twt_str in twts_lst]
    
    #print(df["Split Tweets"]) # Observing output

    return print(df) # NB>>> Remove print()
### END FUNCTION

word_splitter(twitter_df.copy())

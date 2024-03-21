import pandas as pd
import numpy as np


""" ### Data Loading and Preprocessing ### """

# Twitter data
twitter_path = 'python\data\\twitter_nov_2019.csv'
twitter_df = pd.read_csv(twitter_path)


### START FUNCTION
def number_of_tweets_per_day(df):
    """
    Count the number of tweets per day.

    Args:
    df (DataFrame): Twitter dataframe containing the 'Date' and 'Tweets' columns.

    Returns:
    DataFrame: DataFrame with the number of tweets per day.
    """
    # Create a copy of the DataFrame with only the 'Date' and 'Tweets' columns
    df2 = df[['Date', 'Tweets']].copy() 

    # Convert the 'Date' column to a date format
    df2['Date'] = pd.to_datetime(df2['Date']).dt.date 

    # Group the tweets by date and count the number of tweets for each date
    df2 = df2.groupby('Date').size().reset_index(name='Tweets')  

    # Set the 'Date' column as the index of the DataFrame
    df2 = df2.set_index('Date') 
    
    return df2
### END FUNCTION

# Input:
number_of_tweets_per_day(twitter_df.copy())

# Output:
output_data = number_of_tweets_per_day(twitter_df.copy())
print(output_data)


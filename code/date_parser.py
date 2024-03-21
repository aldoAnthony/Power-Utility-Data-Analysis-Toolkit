import pandas as pd
import numpy as np

""" ### Data Loading and Preprocessing ### """
# Twitter data
twitter_path = 'python\data\\twitter_nov_2019.csv'
twitter_df = pd.read_csv(twitter_path)
twitter_df.head()

# Extract dates from Twitter dataframe
dates = twitter_df['Date'].to_list()


### START FUNCTION
def date_parser(dates):
    """
    Extract the date part from a list of date-time strings.

    Args:
    dates (list): A list of date-time strings.

    Returns:
    list: A list containing only the date part of each date-time string.
    """
    # Convert the list of date-time strings to a numpy array
    str_date_time = np.array(dates)
    
    # Split each string into a list of substrings using a space character as a delimiter
    split_str = np.char.split(str_date_time, ' ')

    # Convert the resulting array of lists to a nested list
    split_lst = split_str.tolist()

    # Extract the first element (date part) from each list in the nested list
    date_lst = [time[0] for time in split_lst]

    return date_lst

### END FUNCTION

# Input:
date_parser(dates[-3:])

# Output:
output_data = date_parser(dates[-3:])
print(output_data)

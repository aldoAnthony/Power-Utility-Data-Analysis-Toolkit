import pandas as pd
import numpy as np

""" ### Data Loading and Preprocessing ### """

# Load Twitter data
twitter_path = 'python\data\\twitter_nov_2019.csv'
twitter_df = pd.read_csv(twitter_path)

# Extract dates for Twitter tweets
dates = twitter_df['Date'].to_list()

# Dictionary mapping official municipality Twitter handles to the municipality name
mun_dict = {
    '@CityofCTAlerts': 'Cape Town',
    '@CityPowerJhb': 'Johannesburg',
    '@eThekwiniM': 'eThekwini',
    '@EMMInfo': 'Ekurhuleni',
    '@centlecutility': 'Mangaung',
    '@NMBmunicipality': 'Nelson Mandela Bay',
    '@CityTshwane': 'Tshwane'
}

### START FUNCTION
def extract_municipality_hashtags(df):
    """
    Extract municipality names and hashtags from Twitter dataframe.

    Args:
    df (DataFrame): Twitter dataframe containing the 'Tweets' column.

    Returns:
    DataFrame: Modified dataframe with 'municipality' and 'hashtags' columns added.
    """
    # Function to extract hashtags from a string
    def hashtags(text):
        return list(filter(lambda tag: tag.startswith('#'), text.split()))

    # Add a 'municipality' column and fill it based on matches with Twitter handles
    df['municipality'] = np.nan  # Create a column of NaN values if no match is found
    for key in mun_dict.keys():
        match_key = df['Tweets'].str.contains(key)
        df.loc[match_key, 'municipality'] = mun_dict[key]

    # Extract hashtags from 'Tweets' column and create a new 'hashtags' column
    twt_lst = [hashtags(str1) for str1 in df['Tweets'].values]
    df['hashtags'] = [np.nan if sublst == [] else [tag.lower() for tag in sublst] for sublst in twt_lst]

    return df

### END FUNCTION

# Input:
extract_municipality_hashtags(twitter_df.copy())

# Output:
pd.set_option('display.max_columns', None)# set to display all columns in output
output_data = extract_municipality_hashtags(twitter_df.copy())
print(output_data)

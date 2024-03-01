import pandas as pd
import numpy as np

""" ### Data Loading and Preprocessing ### """

# Electricification by province (EBP) data
#ebp_url = 'https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/electrification_by_province.csv'
ebp_path = '~/Python_Programs/Explorer_Academy_Course_Work/Predicts/electrification_by_province.csv'
ebp_df = pd.read_csv(ebp_path)
#ebp_df = pd.read_csv(ebp_url)

# ~/Python_Programs/Explorer_Academy_Course_Work/Predicts
for col, row in ebp_df.iloc[:,1:].items():
    ebp_df[col] = ebp_df[col].str.replace(',','').astype(int)

#print(ebp_df.head())
ebp_df.head()

# Twitter data
#twitter_url = 'https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/twitter_nov_2019.csv'
twitter_path = '~/Python_Programs/Explorer_Academy_Course_Work/Predicts/twitter_nov_2019.csv'
twitter_df = pd.read_csv(twitter_path)
#print(twitter_df.head())
twitter_df.head()


""" ### Important Variables ### """
# gauteng ebp data as a list
gauteng = ebp_df['Gauteng'].astype(float).to_list()

# dates for twitter tweets
dates = twitter_df['Date'].to_list()

# dictionary mapping official municipality twitter handles to the municipality name
mun_dict = {
    '@CityofCTAlerts' : 'Cape Town',
    '@CityPowerJhb' : 'Johannesburg',
    '@eThekwiniM' : 'eThekwini' ,
    '@EMMInfo' : 'Ekurhuleni',
    '@centlecutility' : 'Mangaung',
    '@NMBmunicipality' : 'Nelson Mandela Bay',
    '@CityTshwane' : 'Tshwane'
}

# dictionary of english stopwords
stop_words_dict = {
    'stopwords':[
        'where', 'done', 'if', 'before', 'll', 'very', 'keep', 'something', 'nothing', 'thereupon', 
        'may', 'why', '’s', 'therefore', 'you', 'with', 'towards', 'make', 'really', 'few', 'former', 
        'during', 'mine', 'do', 'would', 'of', 'off', 'six', 'yourself', 'becoming', 'through', 
        'seeming', 'hence', 'us', 'anywhere', 'regarding', 'whole', 'down', 'seem', 'whereas', 'to', 
        'their', 'various', 'thereafter', '‘d', 'above', 'put', 'sometime', 'moreover', 'whoever', 'although', 
        'at', 'four', 'each', 'among', 'whatever', 'any', 'anyhow', 'herein', 'become', 'last', 'between', 'still', 
        'was', 'almost', 'twelve', 'used', 'who', 'go', 'not', 'enough', 'well', '’ve', 'might', 'see', 'whose', 
        'everywhere', 'yourselves', 'across', 'myself', 'further', 'did', 'then', 'is', 'except', 'up', 'take', 
        'became', 'however', 'many', 'thence', 'onto', '‘m', 'my', 'own', 'must', 'wherein', 'elsewhere', 'behind', 
        'becomes', 'alone', 'due', 'being', 'neither', 'a', 'over', 'beside', 'fifteen', 'meanwhile', 'upon', 'next', 
        'forty', 'what', 'less', 'and', 'please', 'toward', 'about', 'below', 'hereafter', 'whether', 'yet', 'nor', 
        'against', 'whereupon', 'top', 'first', 'three', 'show', 'per', 'five', 'two', 'ourselves', 'whenever', 
        'get', 'thereby', 'noone', 'had', 'now', 'everyone', 'everything', 'nowhere', 'ca', 'though', 'least', 
        'so', 'both', 'otherwise', 'whereby', 'unless', 'somewhere', 'give', 'formerly', '’d', 'under', 
        'while', 'empty', 'doing', 'besides', 'thus', 'this', 'anyone', 'its', 'after', 'bottom', 'call', 
        'n’t', 'name', 'even', 'eleven', 'by', 'from', 'when', 'or', 'anyway', 'how', 'the', 'all', 
        'much', 'another', 'since', 'hundred', 'serious', '‘ve', 'ever', 'out', 'full', 'themselves', 
        'been', 'in', "'d", 'wherever', 'part', 'someone', 'therein', 'can', 'seemed', 'hereby', 'others', 
        "'s", "'re", 'most', 'one', "n't", 'into', 'some', 'will', 'these', 'twenty', 'here', 'as', 'nobody', 
        'also', 'along', 'than', 'anything', 'he', 'there', 'does', 'we', '’ll', 'latterly', 'are', 'ten', 
        'hers', 'should', 'they', '‘s', 'either', 'am', 'be', 'perhaps', '’re', 'only', 'namely', 'sixty', 
        'made', "'m", 'always', 'those', 'have', 'again', 'her', 'once', 'ours', 'herself', 'else', 'has', 'nine', 
        'more', 'sometimes', 'your', 'yours', 'that', 'around', 'his', 'indeed', 'mostly', 'cannot', '‘ll', 'too', 
        'seems', '’m', 'himself', 'latter', 'whither', 'amount', 'other', 'nevertheless', 'whom', 'for', 'somehow', 
        'beforehand', 'just', 'an', 'beyond', 'amongst', 'none', "'ve", 'say', 'via', 'but', 'often', 're', 'our', 
        'because', 'rather', 'using', 'without', 'throughout', 'on', 'she', 'never', 'eight', 'no', 'hereupon', 
        'them', 'whereafter', 'quite', 'which', 'move', 'thru', 'until', 'afterwards', 'fifty', 'i', 'itself', 'n‘t',
        'him', 'could', 'front', 'within', '‘re', 'back', 'such', 'already', 'several', 'side', 'whence', 'me', 
        'same', 'were', 'it', 'every', 'third', 'together'
    ]
}

### START FUNCTION
def extract_municipality_hashtags(df):
    pd.set_option('display.max_columns', None)# set to display all columns in output

    # The "hashtags" function filters out any words in a string that does not start with '#' 
    # and returns a list of the remaining words that are hashtags.
    def hashtags(str):
        return list(filter(lambda tag: tag.startswith('#'), str.split()))

    # muni = df.Tweets.map(mun_dict) # Compares dict key with 'Tweets' column values in 'df' dataframe. If a match is found, dictionary value is put into column
    # df['municipality'] = muni # adds new column to existing dataframe

    df['municipality'] = np.nan # creates a column of NaN values if no match is found
    for key in mun_dict.keys(): # Iterates over the keys in the dictionary and check for matches in the DataFrame    
        match_key = df['Tweets'].str.contains(key) # Create a boolean Series for whether each row in the DataFrame contains the key
        df.loc[match_key, 'municipality'] = mun_dict[key] # Fill in the 'matches' column with the corresponding values for the current key


    twts_arr = list(df['Tweets'].values) # converts 'Tweet 'column into a numpy array then into a list
    #print(twts_arr)# Observing output

    twt_lst = [hashtags(str1) for str1 in twts_arr] # list comprehension 
    #print(twt_lst)# Observing output

    # Code iterates through each sublist in the original list, "twt_lst", 
    # and replaces any empty sublists with the value: np.nan ---> Nan
    # and assigns it as a new column to the "df" data frame named "hashtags"
    df['hashtags'] = [np.nan if sublst == [] else [tag.lower() for tag in sublst] for sublst in twt_lst]
    
    #print(df) # Observing output
    return df
### END FUNCTION

extract_municipality_hashtags(twitter_df.copy())

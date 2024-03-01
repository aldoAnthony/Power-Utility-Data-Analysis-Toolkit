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
def number_of_tweets_per_day(df):
    
    # The ".copy()" method make a copy of the data to avoid modifying the original "df" DataFrame
    df2 = df[['Date', 'Tweets']].copy() 

    df2['Date'] = pd.to_datetime(df2['Date']).dt.date # The "Date" column of "df2" is converted to a ONLY date format using the "pd.to_datetime()" method

    # The ".reset_index method()" is used to reset the index of the resulting DataFrame 
    # and the name parameter is used to set the name of the second column to "Tweets".
    #
    # ".groupby method()" is used to group the tweets by date in "df2", 
    #   and the ".size()" method is used to count the number of tweets for each date.
    df2 = df2.groupby('Date').size().reset_index(name='Tweets')  

    df2 = df2.set_index('Date') # "Date" column is set as the index of the "df2" dataframe
    
    return print(df2) # NB>>> Remove print()
### END FUNCTION

number_of_tweets_per_day(twitter_df.copy())

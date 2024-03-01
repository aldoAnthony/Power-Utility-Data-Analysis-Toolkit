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
def stop_words_remover(df):

    pd.set_option('display.max_columns', None) # Set to display all columns in output

    twt_lst = list(df['Tweets'].values) # Converts 'Tweet 'column into a numpy array then into a list
    #print(twt_lst) # Observing output

    # Code takes a list of strings "twt_lst" and splits each string into a list of strings and assigns it to the variable twt_lst
    twt_lst =  [twt_str.split() for twt_str in twt_lst] 
    #print(twt_lst) # Observing output

    # Extracts the list of stopwords from my_dict and assigns it to the variable stopwords
    stopwords = stop_words_dict['stopwords']

    # Code iterates over every element of "twt_lst". Within this loop, the list comprehension is used
    # to create a new list that contains only those words from the original list that are not in the stopwords list.
    # The condition in the list comprehension checks whether the lowercase version of each word in the current string
    # of twt_lst is not in the stopwords list. If it is, the word is excluded from the new list.
    # The modified list is assigned back to twt_lst to update its contents.
    for i in range(len(twt_lst)):
        twt_lst[i] = [word.lower() for word in twt_lst[i] if word.lower() not in stopwords]
    #print(twt_lst) # Observing output



    df["Without Stop Words"] = twt_lst # assign "twt_lst" as a column in the "df" dataframe named "Without Stop Words"

    #print(df)
    return print(df) # NB>>> Remove print()
### END FUNCTION 

stop_words_remover(twitter_df.copy())

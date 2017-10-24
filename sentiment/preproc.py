"""
This script need to pre processing raw data  from twitter
"""

import pandas as pd
import os
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split

DATA_PATH = "./data/tweets/Sentiment_Analysis_Dataset.csv"
en_stopwords = set(stopwords.words("english"))


def tokenize(text):
    tknzr = RegexpTokenizer(r'\w+')
    return tknzr.tokenize(text)


def stem(text):
    st = SnowballStemmer('english')
    return [st.stem(w) for w in text]


def preproc_sentence(sent):
    tokens_sent = tokenize(sent)
    stem_tokens = stem(tokens_sent)
    filtered_words = list(filter(lambda word: word not in en_stopwords, stem_tokens))
    return ' '.join(filtered_words)


path_to_save = "./data/tweets"
full_data = pd.read_csv(DATA_PATH, error_bad_lines=False)
df = full_data[['SentimentText', 'Sentiment']]
df['SentimentText'] = df.loc[:,'SentimentText'].apply(preproc_sentence)

train, test = train_test_split(df, test_size=0.15, random_state=1)
valid = train[:100]

valid.to_csv(os.path.join(path_to_save,'valid.txt'),  header=True, index=False, sep='\t', mode='w')
test.to_csv(os.path.join(path_to_save,'test.txt'),  header=True, index=False, sep='\t', mode='w')
train.to_csv(os.path.join(path_to_save,'test.txt'),  header=True, index=False, sep='\t', mode='w')

print("Finished cleaning the data")
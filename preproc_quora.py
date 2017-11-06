import pandas as pd
import string

import os
dir_path ="./data/quora/"


def create_data(path_to_data):
    raw_file_name = os.path.join(path_to_data, "quora_duplicate_questions.tsv")
    data = pd.read_csv(raw_file_name)
    # drop rows with null value
    data.dropna(inplace=True)
    # make columns of lower cased words
    data["cleaned_q1"] = data.question1.str.lower()
    data["cleaned_q2"] = data.question2.str.lower()
    # remove punctuation from lower-cased words columns
    data['cleaned_q1'] = data['cleaned_q1'].apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
    data['cleaned_q2'] = data['cleaned_q2'].apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
    # remove the character "\n", which messes up the line delimiters in txt file
    data['cleaned_q2'] = data['cleaned_q2'].apply(lambda x: x+'.')
    data['cleaned_q1'] = data['cleaned_q1'].apply(lambda x: x+'.')

    data["cleaned_q1"] = data['cleaned_q1'].str.replace("\n", ".")
    data["cleaned_q2"] = data['cleaned_q2'].str.replace("\n", ".")
    # shuffle data before writing to file - this way random sample can be taken from file
    data = data.sample(frac=1)
    df = pd.DataFrame()
    df['sentence1'] = data["cleaned_q1"]
    df['sentence2'] = data["cleaned_q2"]
    df['duplicate'] = data["is_duplicate"]

    idx_to_split = int(0.002*len(data))
    print(df.head())
    train = df.iloc[idx_to_split:]
    test = df.iloc[:idx_to_split]
    valid = train[:idx_to_split]
    train.to_csv(path_to_data+"train.txt", sep='\t',  index=False)
    test.to_csv(path_to_data+"test.txt", sep='\t', index=False)
    valid.to_csv(path_to_data+"valid.txt", sep='\t',  index=False)


if __name__ == '__main__':
    create_data(dir_path)
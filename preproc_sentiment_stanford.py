import os
import pandas as pd
from six.moves import cPickle
from sklearn.utils import shuffle

"""
This module read all data from stanford Sentiment tree bank,
convert them into comfortable form
"""

dir_path = "./data/stanfordSentimentTreebank/"
def read_splitlabel():

    ifname = os.path.join(dir_path ,'datasetSplit.txt')
    lines = open(ifname, 'r').read().split('\n')

    splitlabels = []
    for line in lines[1:]:
        params = line.split(',')
        if len(params) == 2:
            splitlabels.append(int(params[1]))

    return splitlabels


def read_text():
    ifname = os.path.join(dir_path ,'SOStr.txt')
    lines = open(ifname, 'r').read().split('\n')

    texts = []
    for line in lines:
        params = line.split('|')
        if len(params) > 1:
            text = ' '.join(params)
            texts.append(text)

    return texts


def read_sentiscore():
    ifname = os.path.join(dir_path, 'sentiment_labels.txt')
    lines = open(ifname, 'r').read().split('\n')

    sentiscores = []
    for line in lines[1:]:
        params = line.split('|')
        if len(params) == 2:
            sentiscores.append(float(params[1]))
    return sentiscores


def read_phrase_ids():
    ifname = os.path.join(dir_path, 'dictionary.txt')
    lines = open(ifname, 'r').read().split('\n')

    phrase_ids = {}
    for line in lines:
        params = line.split('|')
        if len(params) == 2:
            phrase_ids[params[0]] = int(params[1])

    return phrase_ids


def prepare_dataset():
    texts = read_text()
    split_labels_table = read_splitlabel()
    senti_label_table = read_sentiscore()
    phrase_ids = read_phrase_ids()

    train_text = []
    train_label = []

    valid_text = []
    valid_label = []

    test_text = []
    test_label = []

    n_sample = len(texts)
    if n_sample == len(split_labels_table) and len(senti_label_table) == len(phrase_ids):
        print('%d samples' % (n_sample))
    else:
        print("Reading failed")

    for i, didx in enumerate(split_labels_table):
        if didx == 1:
            list_text = train_text
            list_label = train_label
        elif didx == 3:
            list_text = valid_text
            list_label = valid_label
        elif didx == 2:
            list_text = test_text
            list_label = test_label
        lable = senti_label_table[phrase_ids[texts[i]]]
        if (lable < 0.6) & (lable > 0.4):
            continue
        list_text.append(texts[i])
        list_label.append(senti_label_table[phrase_ids[texts[i]]])

    def reduce_sentiment(labels):
        y = []
        for l in labels:
            if l <= 0.5:
                y.append(0)
            else:
                y.append(1)

        return y
    train_dataset = shuffle(pd.DataFrame({"SentimentText":train_text, "Sentiment":reduce_sentiment(train_label)}))
    test_dataset = shuffle(pd.DataFrame({"SentimentText":test_text, "Sentiment": reduce_sentiment(test_label)}))
    valid_dataset = shuffle(pd.DataFrame({"SentimentText":valid_text, "Sentiment": reduce_sentiment(valid_label)}))

    train_dataset.to_csv(os.path.join(dir_path,"train.txt"), index=False)
    valid_dataset.to_csv(os.path.join(dir_path,"valid.txt"), index=False)
    test_dataset.to_csv(os.path.join(dir_path,"test.txt"), index=False)


if __name__ == '__main__':
    prepare_dataset()

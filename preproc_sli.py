import pandas as pd
import json
import os
dir_path ="./data/snli_1.0/"


def read_data(train_filename):
    (train_dataset, train_sentence1, train_sentence2, train_labels) = loadDataset(train_filename, 10000)
    #train_df[['sentence1', 'sentence2', 'gold_label']]
    df = pd.DataFrame()
    df['sentence1'] = train_sentence1
    df['sentence2'] = train_sentence2
    df['gold_label'] = train_labels
    return  df


def loadDataset(filename, size=-1):
    label_category = {
        'contradiction': 0,
        'entailment': 1
#        'neutral': 2
    }
    dataset = []
    sentence1 = []
    sentence2 = []
    labels = []
    def labelize(labels):
        y = []
        for l in labels:
            if l =='neutral':
                y.append(0)
            elif l =='entailment':
                y.append(1)
            else:
                y.append(2)
        return y

    with open(filename, 'r') as f:
        i = 0
        not_found = 0
        for line in f:
            row = json.loads(line)
            if (i > 10) and (sentence1[-1] == row['sentence1'].strip()):
                continue
            if size == -1 or i < size:
                dataset.append(row)
                label = row['gold_label'].strip()
                if label in label_category:
                    sentence1.append(row['sentence1'].strip())
                    sentence2.append(row['sentence2'].strip())
                    labels.append(label_category[label])
                    i += 1
                else:
                    not_found += 1
            else:
                break
        if not_found > 0:
            print('Label not recognized %d' % not_found)

    return (dataset, sentence1, sentence2, labels)

if __name__ == '__main__':
    df = read_data("./data/snli_1.0/snli_1.0_train.jsonl")
    train_df = df[:8000]
    test_df = df[8000:8500]
    print(test_df.columns)
    valid_df = df[8500:8700]
    train_df.to_csv(os.path.join(dir_path, "train.txt"), index=False)
    valid_df.to_csv(os.path.join(dir_path, "valid.txt"), index=False)
    test_df.to_csv(os.path.join(dir_path, "test.txt"), index=False)
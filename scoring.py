import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mean_squared_error, roc_auc_score
from jinja2 import Template
from scipy import spatial
import click
from tqdm import tqdm
import json
import os
from sample import sample_multi


@click.command()
@click.option('--data-dir', default='/data/yanews/docs')
@click.option('--template', default='templates/vis.html')
@click.option('--output', default='output.html')
@click.option('--save-dir', default='save')
def main(data_dir, template, output, save_dir):
    clusters = {}
    clustered = {}
    print('Load clustermap')
    with open(os.path.join(data_dir, 'SBEADS.resC'), 'r') as fin:
        for line in tqdm(fin):
            tab = line.index('\t')
            cluster_id = int(line[:tab])
            ids = [int(t) for t in line[tab + 1:].split()]
            for i in ids:
                clustered[i] = cluster_id
            clusters[cluster_id] = ids

    print('Load messages')
    data = []
    with open(os.path.join(data_dir, 'docs.out'), 'r') as fin:
        for line in tqdm(fin):
            # take care about escape symbols
            filtered = line.replace(r"\'", "'").replace('\\', '/')
            try:
                entry = json.loads(filtered)
                id = int(entry.get('id'))
                title = entry.get('title')
                data.append((id, title))
            except:
                print(filtered)

    word2vectors = sample_multi(save_dir, [t[1] for t in data])

    X = word2vectors

    print('Compute natural w2v clusterization quality.')

    # ATTENTION: very dirty code, just let it works
    n = 10000
    clust_numbers = list(clusters.keys())
    selected = np.random.choice(clust_numbers, n)

    res = []
    # not sure if they are continuous:
    indexes = {t[0]: i for (i, t) in enumerate(data)}
    misscounter = 0
    for i in tqdm(selected):
        tmp = []
        for j in clusters[i]:
            if j in indexes:
                tmp.append(indexes[j])
            else:
                misscounter += 1

        if len(tmp) < 2:
            # bad, let it go
            # print('Pass')
            continue
        # get pair
        pair = np.random.choice(tmp, 2)
        one = X[pair[0], :]
        two = X[pair[1], :]
        sim = 1.0 - spatial.distance.cosine(one, two)
        if np.isnan(sim) or np.isinf(sim):
            sim = 0.0
        res.append((1.0, sim))

        # (try to) get nonpair
        one = np.random.choice(tmp, 1)
        two = np.random.random_integers(0, len(X) - 1, 1)
        gnd = 1.0 * (two in tmp)
        one = X[one, :]
        two = X[two, :]
        sim = 1.0 - spatial.distance.cosine(one, two)
        if np.isnan(sim) or np.isinf(sim):
            sim = 0.0

        res.append((gnd, sim))

    res = np.array(res)
    print("Classes ratio:\t%.3f" % (sum(res[:, 0]) / len(res)))
    print("MSE:\t\t%.3f" % mean_squared_error(res[:, 0], res[:, 1]))
    print("AUC:\t\t%.3f" % roc_auc_score(res[:, 0], res[:, 1]))
    # last result was
    # Classes ratio:	0.500
    # MSE:		0.106
    # AUC:		0.968
    # End of ATTENTION

    labels = np.array([clustered.get(t[0], -1) for t in data])
    score = silhouette_score(X, labels, sample_size=1000)
    # it gives me about 0.77
    print('Natural w2v silhouette_score is {}'.format(score))

    idx = (labels > -1)

    score = silhouette_score(X[idx], labels[idx], sample_size=1000)
    # it gives me about 0.87
    print('Without unclustered samples score is {}'.format(score))

    # Preparation for visualization. Unfinished yet
    print('Compute 2d projection')
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    print('Save the data')

    repack = []
    with open('data.csv', 'w') as fout:
        for i, x in zip([t[0] for t in data], X2):
            q = (x[0], x[1], i, clustered.get(i, -1))
            fout.write('{:.2f},{:.2f},{},{}\n'.format(*q))
            repack.append(q)
    repack = json.dumps(repack)

    # Experiment with coarsed coordinates:
    d1 = len(set(['{:.1f},{:.1f}'.format(x[0], x[1]) for x in X2]))
    d2 = len(set(['{:.2f},{:.2f}'.format(x[0], x[1]) for x in X2]))
    d3 = len(set(['{:.3f},{:.3f}'.format(x[0], x[1]) for x in X2]))
    print('We can coarse the data: ')
    print(d1)
    print(d2)
    print(d3)

    with open(template, 'r') as fin:
        page = Template(fin.read()).render(data=repack)
        with open(output, 'w') as fout:
            fout.write(page)

    with open('labels.csv', 'w') as fout:
        for i, label in data:
            fout.write('{}\t{}\n'.format(i, label))

    print('Done')

if __name__ == "__main__":
    main()
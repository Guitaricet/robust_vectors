import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--result", type=str, default='results.txt', help="path to result")


args = parser.parse_args()

path_to_result = args.result

types = {"random": {}, "word2vec": {}, "robust": {}}
with open(path_to_result) as f:
    for line in f:
        parts = line.strip().split(",")
        print(parts)
        noise = float(parts[1])
        if noise not in types[parts[0]]:
            types[parts[0]][noise] = []
        types[parts[0]][noise].append(float(parts[2]))
types2 = {"random": [], "word2vec": [], "robust": []}
for t in types:
    for n in types[t]:
        mi = min(types[t][n])
        ma = max(types[t][n])
        me = np.mean(types[t][n])
        st = sem(types[t][n])
        types2[t].append((n, me, st))

# x2 = sorted(types2["word2vec"], key=lambda x: x[0])
# x2err = [x[2] for x in x2]
# x2 = [x[1] for x in x2]

x1 = sorted(types2["robust"], key=lambda x: x[0])
y = [x[0] for x in x1]
xerr = [x[2] for x in x1]
x1 = [x[1] for x in x1]

#x3 = [types2["random"][0][1]] * 11

plt.figure()
p1, = plt.plot(y, x1, 'g')
#p2, = plt.plot(y, x2, "r")
#p3, = plt.plot(y, x3, "b")
plt.title("Quality against noise")
#plt.legend([p1, p2, p3], ['Standard Word2Vec', 'Robust Word2Vec', 'Random'])
plt.xlabel('noise level')
plt.ylabel('ROC AUC')
plt.show()

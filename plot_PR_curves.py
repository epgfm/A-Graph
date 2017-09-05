#! /usr/bin/python -u

from misc import *

from classif_dump_results import compute_results, compute_results_with_threshold


import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

import numpy as np


fig = plt.figure()

# let's draw PR curve for clf 0
all_pre = []
all_rec = []

gen = ranges_gen()

for i in range(10):

    train, test = gen.next()

    clf = zload("models/%s.pkl.gz" % i)

    scores_rec = []
    scores_pre = []
    t = 0.93
    while t > 0.0:

        rec, pre, fp, fn, tp, tn, acc = compute_results_with_threshold(
                                        i, clf, test, "", t)
        scores_rec.append(rec)
        scores_pre.append(pre)

        print t
        t -= 0.02
    
    plt.plot(scores_rec, scores_pre, color="#666666", alpha = 0.2)
    all_pre.append(scores_pre)
    all_rec.append(scores_rec)


avg_pre = []
avg_rec = []
for i in range(len(all_pre[0])):
    sum_pre = 0
    sum_rec = 0
    for j in range(len(all_pre)):
        sum_pre += all_pre[j][i]
        sum_rec += all_rec[j][i]
    avg_pre.append(sum_pre / len(all_pre))
    avg_rec.append(sum_rec / len(all_rec))

plt.plot(avg_rec, avg_pre, color = 'r')


plt.ylabel('Precision')
plt.xlabel('Recall')

plt.show()

fig.savefig("all_pr.pdf",  bbox_inches='tight')

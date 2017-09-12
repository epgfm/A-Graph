#! /usr/bin/python -u

import os, sys, pickle, random, glob, gzip
import argparse as ap

from compute_centralities import *
from misc import *

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV




def load_labels(parts):
    labels = {}
    for p in parts:
        X, part_labels = load_full_comments(p)
        labels[p] = part_labels
    return labels


def strip_indices(centralities, indices):
    i = 0
    out = []
    for c in centralities:
        if i in indices:
            i += 1
            continue
        else:
            out.append(c)
            i += 1
    return out


if __name__ == '__main__':

    p = ap.ArgumentParser()

    args = p.parse_args()

    g = ranges_gen()

    for j in range(10):

        train, test = g.next()

        res_train = zload("features/centralities-%s-train-noscale-multi.pkl.gz" % j)
        centralities = []
        for r in res_train:
            fns = r[0]
            p, i = fns[0].split("/")[1:3]
            p, i = int(p), int(i)
            centralities.append(strip_indices(r[1], []))

        

        ordered_labels = load_labels(train)
        labels = []
        for r in res_train:
            fns = r[0]
            p, i = fns[0].split("/")[1:3]
            p, i = int(p), int(i)
            labels.append(ordered_labels[p][i])

        print len(centralities[0])

        print len(centralities)


        scaler = StandardScaler().fit(centralities)
        centralities = scaler.transform(centralities)


        clf = SVC(class_weight='balanced')
        clf = CalibratedClassifierCV(clf)
        clf.fit(centralities, labels)
        print os.getcwd()
        zdump(clf, "models/%s.pkl.gz" % j)
        zdump(scaler, "models/%s.scaler.pkl.gz" % j)





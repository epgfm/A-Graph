#! /usr/bin/env python

import os, sys, pickle, random, glob, gzip
import argparse as ap


from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from misc import *
from sklearn.ensemble import ExtraTreesClassifier
from compute_centralities import get_feature_names
from compute_centralities import load_full_comments
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt


def get_indices_from_names(names, fmap):
    return [fmap[n] for n in names]


def get_nbest_names(fi, n):
    ''' (list) -> list
    given a sorted list fi, return n best names
    '''
    return [n[1] for n in fi[:n]]


def load_labels(parts):
    labels = {}
    for p in parts:
        X, part_labels = load_full_comments(p)
        labels[p] = part_labels
    return labels



def load_train_shit(i, parts):
    res_test = zload("features/centralities-%s-train-noscale-multi.pkl.gz" % i)
    centralities = []
    for r in res_test:
        fns = r[0]
        p, i = fns[0].split("/")[1:3]
        p, i = int(p), int(i)
        centralities.append(r[1])
    ordered_labels = load_labels(parts)
    labels = []
    for r in res_test:
        fns = r[0]
        p, i = fns[0].split("/")[1:3]
        p, i = int(p), int(i)
      # print p, i
        labels.append(ordered_labels[p][i])
    return centralities, labels



def load_test_shit(i, parts):
    res_test = zload("features/centralities-%s-test-noscale-multi.pkl.gz" % i)
    centralities = []
    for r in res_test:
        fns = r[0]
        p, i = fns[0].split("/")[1:3]
        p, i = int(p), int(i)
        centralities.append(r[1])
    ordered_labels = load_labels(parts)
    labels = []
    for r in res_test:
        fns = r[0]
        p, i = fns[0].split("/")[1:3]
        p, i = int(p), int(i)
      # print p, i
        labels.append(ordered_labels[p][i])
    return centralities, labels



def get_fnames():
    fnames = ["full-" + name for name in get_feature_names()] + \
             ["before-" + name for name in get_feature_names()] + \
             ["after-" + name for name in get_feature_names()]
    return fnames

def get_fmap():
    fmap = {}
    fnames = get_fnames()
    i = 0
    for i in range(len(fnames)):
        fmap[fnames[i]] = i
        i += 1
    return fmap


def get_feature_importance(i, train, test):
    
    X, y = load_train_shit(i, train)
    Xt, yt = load_test_shit(i, test)
    X.extend(Xt)
    y.extend(yt)

    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)


    fnames = ["full-" + name for name in get_feature_names()] + \
             ["before-" + name for name in get_feature_names()] + \
             ["after-" + name for name in get_feature_names()]
              

    out = []
    i = 0
    for fi in clf.feature_importances_:
        out.append((fi, fnames[i]))
        i += 1
    return out




def get_model_from_features(prefix, i, indices):

    X, y = load_train_shit(i, train)
    X = select_features(X, indices)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    clf = LinearSVC(class_weight='balanced')
  # clf = CalibratedClassifierCV(clf) # So we can use predict_prioba
    clf.fit(X, y)

    return clf, scaler


def select_features(feature_arrays, indices):
    ''' (list of list of float, list of int) -> list of list of float

    Return a new feature arrays with only the features specified in indices.

    '''
    out = []
    for fa in feature_arrays:
        sub = []
        for i in indices:
            sub.append(fa[i])
        out.append(list(sub))
    return out


def compute_results_from_features(i, test, clf, indices, scaller):
   
    X, y = load_test_shit(i, test)
    X = select_features(X, indices)

    X = scaller.transform(X)

    res = clf.predict(X)

    ok = 0
    fp, fn = 0, 0
    tp, tn = 0, 0
    nAbuses = 0
    for r in range(len(res)):
        if res[r] == y[r]:
            ok += 1
        if y[r] == 1:
            if res[r] == 1:
                tp += 1
            if res[r] == 0:
                fn += 1
            nAbuses += 1
        if y[r] == 0:
            if res[r] == 1:
                fp += 1
            if res[r] == 0:
                tn += 1

    rec = tp / float(nAbuses)
    try:
        pre = tp / float(tp + fp)
    except:
        pre = 1
    return rec, pre, fp, fn, tp, tn



if __name__ == '__main__':

    p = ap.ArgumentParser()

    args = p.parse_args()

    

    fmap = get_fmap()

    for j in range(200):

        gen = ranges_gen()
        train, test = gen.next()

        fi = get_feature_importance(0, train, test)

        fi.sort(key = lambda x: -x[0])


        f1s = []
        ACCs = []
        n = len(fi)

        while n > 0:

            best_names = get_nbest_names(fi, n)
            indices = get_indices_from_names(best_names, fmap)
            print indices

            scores = {k : [] for k in ['rec', 'pre', 'acc', 'fp', 'fn', 'tp', 'tn']}
            gen = ranges_gen()
            for i in range(10):
                train, test = gen.next()
                clf, scaller = get_model_from_features("", i, indices)
                rec, pre, fp, fn, tp, tn = compute_results_from_features(i, test, clf, indices, scaller)
                scores['rec'].append(rec)
                scores['pre'].append(pre)
                leny = fp + fn + tp + tn
                scores['fp'].append(fp / float(leny))
                scores['fn'].append(fn / float(leny))
                scores['tp'].append(tp / float(leny))
                scores['tn'].append(tn / float(leny))
                scores['acc'].append((tp+tn) / float(leny))
                


            rec = sum(scores['rec']) / len(scores['rec'])
            pre = sum(scores['pre']) / len(scores['pre'])
            afp = sum(scores['fp']) / len(scores['fp'])
            afn = sum(scores['fn']) / len(scores['fn'])
            atp = sum(scores['tp']) / len(scores['tp'])
            atn = sum(scores['tn']) / len(scores['tn'])
            aacc = sum(scores['acc']) / len(scores['acc'])


            print n, " & ".join(["%0.4f" % v for v in [aacc, rec, pre, afp, afn, atp, atn]])
            f1 = 2 * (pre * rec) / float(pre + rec)
            f1s.append(f1)
            ACCs.append(aacc)
            n -= 1



        plt.plot(f1s)
        plt.ylabel('F1-Score')
        plt.xlabel('Number Of Features Removed')

        zdump((f1s, ACCs, fi), "fi-F-%s-DUAL.pkl.gz" % j)
        plt.savefig("test-F-%s-DUAL.png" % j)




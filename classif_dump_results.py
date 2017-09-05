#! /usr/bin/python -u

import os, sys, pickle, random, glob, gzip
import argparse as ap
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.preprocessing import StandardScaler

from misc import *


def comments_from_file(fileName):
    ''' Load and return messages in fileName '''
    content = zload(fileName)
    comments = [c for c in content]
    return comments

def load_full_comments(p):
    X, y = [], []
    zerosFile = glob.glob("../../../10-Split/*0.*-%s*" % p)[0]
    zeros = comments_from_file(zerosFile)
    onesFile = glob.glob("../../../10-Split/*1.*-%s*" % p)[0]
    ones = comments_from_file(onesFile)
    y.extend([0 for v in range(len(zeros))])
    y.extend([1 for v in range(len(ones))])
    X.extend(zeros)
    X.extend(ones)
    return X, y


def load_labels(parts):
    labels = {}
    for p in parts:
        X, part_labels = load_full_comments(p)
        labels[p] = part_labels
    return labels



def strip_indices(centralities, indices):
    k = 0
    out = []
    for c in centralities:
        if k in indices:
            k += 1
            continue
        else:
            out.append(c)
            k += 1
        
    return out


def compute_results(j, clf, test, prefix = ""):

    res_test = zload("features/centralities-%s-test-noscale-multi.pkl.gz" % j)
    centralities = []
    for r in res_test:
        fns = r[0]
        p, i = fns[0].split("/")[1:3]
        p, i = int(p), int(i)
        centralities.append(strip_indices(r[1], []))

    
    scaler = zload("models/%i.scaler.pkl.gz" % j)
    centralities = scaler.transform(centralities)


    ordered_labels = load_labels(test)
    labels = []
    for r in res_test:
        fns = r[0]
        p, i = fns[0].split("/")[1:3]
        p, i = int(p), int(i)
      # print p, i
        labels.append(ordered_labels[p][i])

    X = centralities
    y = labels


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
    pre = tp / float(tp + fp)
    acc = ok / float(len(res))

    return rec, pre, fp, fn, tp, tn, acc





def compute_results_with_threshold(j, clf, test, prefix = "", threshold = 0.5):

    res_test = zload("features/centralities-%s-test-noscale-multi.pkl.gz" % j)
    centralities = []
    for r in res_test:
        fns = r[0]
        p, i = fns[0].split("/")[1:3]
        p, i = int(p), int(i)
        centralities.append(strip_indices(r[1], []))

    
    scaler = zload("models/%i.scaler.pkl.gz" % j)
    centralities = scaler.transform(centralities)

    ordered_labels = load_labels(test)
    labels = []
    for r in res_test:
        fns = r[0]
        p, i = fns[0].split("/")[1:3]
        p, i = int(p), int(i)
      # print p, i
        labels.append(ordered_labels[p][i])

    X = centralities
    y = labels


    prob_pos = clf.predict_proba(X)


    res = []
    for p in prob_pos:
        if p[1] > threshold:
            res.append(1)
        else:
            res.append(0)

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
        pre = 0
    acc = ok / float(len(res))
    return rec, pre, fp, fn, tp, tn, acc









def compute_10fold_results(prefix = ""):

    scores = {k : [] for k in ['rec', 'fp', 'fn', 'tp', 'tn', 'acc']}

    g = ranges_gen()

    for it in range(10):
        train, test = g.next()

        clf = zload("models/%s.pkl.gz" % it)
        rec, pre, fp, fn, tp, tn, acc = compute_results(it, clf, test, prefix)

        scores['rec'].append(rec)
        leny = fp + fn + tp + tn
        scores['fp'].append(fp / float(leny))
        scores['fn'].append(fn / float(leny))
        scores['tp'].append(tp / float(leny))
        scores['tn'].append(tn / float(leny))
        scores['acc'].append(acc)
        if not args.compact:
            print pre, rec, acc, leny, tn+fp, tp+fn


    rec = sum(scores['rec']) / len(scores['rec'])
    afp = sum(scores['fp']) / len(scores['fp'])
    afn = sum(scores['fn']) / len(scores['fn'])
    atp = sum(scores['tp']) / len(scores['tp'])
    atn = sum(scores['tn']) / len(scores['tn'])
    aacc = sum(scores['acc']) / len(scores['acc'])
    pre = atp / (atp + afp)
    return rec, pre, afp, afn, atp, atn, aacc


if __name__ == '__main__':

    p = ap.ArgumentParser() 
    p.add_argument("--compact", action="store_true")
    args = p.parse_args()


    if not args.compact:
        print " & ".join(["Pre", "Rec", "Acc", "test_size"])

    rec, pre, afp, afn, atp, atn, acc = compute_10fold_results()
    f1 = 2 * (pre * rec) / (pre + rec)

    if args.compact:
        print " & ".join(["Pre", "Rec", "F1"])
        print " & ".join(["%0.4f" % v for v in [pre, rec, f1]])
    else:

        print "Average Precision: %0.4f" % pre,
        print "Average Recall: %0.4f" % rec,
        print "Average Accuracy: %0.4f" % acc
        print "Average FP: %s" % afp
        print "Average FN: %s" % afn
        print "Average TP: %s" % atp
        print "Average TN: %s" % atn






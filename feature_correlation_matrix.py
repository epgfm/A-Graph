#! /usr/bin/python -u

import os, sys, pickle, random, glob, gzip
import argparse as ap

from misc import *
from compute_centralities import get_feature_names
from scipy.stats.stats import pearsonr

if __name__ == '__main__':

    p = ap.ArgumentParser()

    args = p.parse_args()


    features = zload("features/centralities-0-train-noscale-multi.pkl.gz")


    feature_values = {}
    feature_names = ["full-" + fn for fn in get_feature_names()] + \
                    ["before-" + fn for fn in get_feature_names()] + \
                    ["after-" + fn for fn in get_feature_names()]



    labels = []
    for f in features:
        filename, fvs, label = f
        for i in range(len(fvs)):
            if feature_names[i] in feature_values:
                feature_values[feature_names[i]].append(fvs[i])
            else:
                feature_values[feature_names[i]] = [fvs[i]]
        labels.append(label)

    nfeat = len(feature_names)
    print ";",
    for f in feature_names:
        print f, ";",
    print
    for i in range(nfeat):
        print feature_names[i], ";",
        for j in range(nfeat):
            corr, pvalue = pearsonr(feature_values[feature_names[i]],
                                    feature_values[feature_names[j]])
            if pvalue < 0.01:
                print corr, ";",
            else:
                print "NOCORR ;",
        print


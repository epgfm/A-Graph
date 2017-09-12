#! /usr/bin/python -u

import os, sys, pickle, random, glob, gzip
import argparse as ap

from misc import *
from compute_centralities import get_feature_names
from scipy.stats.stats import pearsonr


def class_averages(feature_values, labels):
    c_0_values = []
    c_1_values = []
    for v, l in zip(feature_values, labels):
        if int(l) == 0:
            c_0_values.append(v)
        else:
            c_1_values.append(v)
    avg_0 = sum(c_0_values) / float(len(c_0_values))
    avg_1 = sum(c_1_values) / float(len(c_1_values))
    return avg_0, avg_1

if __name__ == '__main__':

    p = ap.ArgumentParser()

    args = p.parse_args()


    features = zload("features/centralities-0-train-noscale-multi.pkl.gz")


    feature_values = {}
    feature_names = ["full-" + fn for fn in get_feature_names()] + \
                    ["before-" + fn for fn in get_feature_names()] + \
                    ["after-" + fn for fn in get_feature_names()]

    print feature_names

    labels = []
    for f in features:
        filename, fvs, label = f
        for i in range(len(fvs)):
            if feature_names[i] in feature_values:
                feature_values[feature_names[i]].append(fvs[i])
            else:
                feature_values[feature_names[i]] = [fvs[i]]
        labels.append(label)



    target_features = [
        'full-average_betweeness_centrality',
        'before-average_coreness',
        'after-edge_count',
        'after-density',
        'full-Hub',
        'after-Degree_Centrality',
        'before-edge_count',
        'full-average_eccentricity',
        'before-average_eigenvector_centrality',
        'full-Eccentricity',
    ]

    print "Averages per class (class_0, class_1 aka abuse)"
    for t in target_features:
        print "[%s]" % t, class_averages(feature_values[t], labels)

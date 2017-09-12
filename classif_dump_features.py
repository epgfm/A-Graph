#! /usr/bin/python -u

import os, sys, pickle, random, glob, gzip
import argparse as ap

from compute_centralities import *
from misc import *


from sklearn import preprocessing

if __name__ == '__main__':

    p = ap.ArgumentParser()

    args = p.parse_args()

    generator = ranges_gen()

    for i in range(10):

        train, test = generator.next()

        centralities = []
        labels = []

        for p in train:
            targets = glob.glob("../%s/*/full.graphml" % p)
            for t in targets:
                g = igraph.Graph.Read_GraphML(t)
                target_uid = g['target_uid']
                filenames = target_graphml_filenames(t)
                centralities.append(get_all_centralities(target_uid, filenames))
                label = g['label']
                # Load graphml in igraph
                labels.append(label)
                print ".",
        print


        print "Dumping:", "features/centralities-%s-train.pkl.gz" % i
        zdump((centralities, labels), "features/centralities-%s-train-noscale.pkl.gz" % i)

        centralities = []

        for p in test:
            targets = glob.glob("../%s/*/full.graphml" % p)
            for t in targets:
                g = igraph.Graph.Read_GraphML(t)
                target_uid = g['target_uid']
                filenames = target_graphml_filenames(t)
                centralities.append(get_all_centralities(target_uid, filenames))
                label = g['label']
                # Load graphml in igraph
                labels.append(label)
                print ".",


        print "Dumping:", "features/centralities-%s-test.pkl.gz" % i
        zdump((centralities, labels), "features/centralities-%s-testN.pkl.gz" % i)







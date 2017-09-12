#! /usr/bin/python -u

import os, sys, pickle, random, glob, gzip
import argparse as ap

from compute_centralities import *
from misc import *


from sklearn import preprocessing

from multiprocessing import Pool


def load_all_jobs(p):
    targets = glob.glob("../%s/*/full.graphml" % p)
    out = []
    for t in targets:
        g = igraph.Graph.Read_GraphML(t)
        target_uid = g['target_uid']
        label = g['label']
        filenames = target_graphml_filenames(t)
        out.append((target_uid, filenames, label))
    return out




def run_all_jobs(jobs):
    pool = Pool(processes=48)
    res = []
    res_l = []
    res_fns = []
    for j in jobs:
        params = (j[0], j[1])
        res_fns.append(j[1])
        res_l.append(j[2])
        res.append(pool.apply_async(get_all_centralities, params))
    res_v = []
    for r in res:
        res_v.append(r.get())
        print ".",

    out = []
    for i in range(len(res_l)):
        out.append((res_fns[i], res_v[i], res_l[i]))
    return out


def make_res_array(train_res, test_res):
    out = []
    for r in train_res:
        out.append(r)
    for r in test_res:
        out.append(r)
    return out


def get_saved_res(target_fns, res_array):
    fns = [r[0] for r in res_array]
    return res_array[fns.index(target_fns)]


def get_res_from_array(jobs, res_array):
    out = []
    for j in jobs:
        out.append(get_saved_res(j[1], res_array))
    return out



if __name__ == '__main__':

    p = ap.ArgumentParser()

    args = p.parse_args()

    res_array = None

    g = ranges_gen()
    for i in range(10):
        train, test = g.next()

        if res_array == None:

            train_res = []
            for p in train:
                jobs = load_all_jobs(p)
                res = run_all_jobs(jobs)
                for r in res:
                    train_res.append(r)
            print "Dumping:", "features/centralities-%s-train-noscale-multi.pkl.gz" % i
            zdump(train_res, "features/centralities-%s-train-noscale-multi.pkl.gz" % i)

            test_res = []
            for p in test:
                jobs = load_all_jobs(p)
                res = run_all_jobs(jobs)
                for r in res:
                    test_res.append(r)
            print "Dumping:", "features/centralities-%s-test-noscale-multi.pkl.gz" % i
            zdump(test_res, "features/centralities-%s-test-noscale-multi.pkl.gz" % i)


            res_array = make_res_array(train_res, test_res)

        else:
            
            train_res = []
            for p in train:
                jobs = load_all_jobs(p)
                res = get_res_from_array(jobs, res_array)
                for r in res:
                    train_res.append(r)
            print "Dumping:", "features/centralities-%s-train-noscale-multi.pkl.gz" % i
            zdump(train_res, "features/centralities-%s-train-noscale-multi.pkl.gz" % i)


            test_res = []
            for p in test:
                jobs = load_all_jobs(p)
                res = get_res_from_array(jobs, res_array)
                for r in res:
                    test_res.append(r)
            print "Dumping:", "features/centralities-%s-test-noscale-multi.pkl.gz" % i
            zdump(test_res, "features/centralities-%s-test-noscale-multi.pkl.gz" % i)







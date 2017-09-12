#! /usr/bin/python -u

import os, sys, pickle, random, glob, gzip, string
import argparse as ap


from misc import zload, zdump

'''
    So if we've got a feature that never triggers a performance drop when we
    remove it then it's safe to say it's a useless feature?

    Let's see if there's any.

    Note: because of some fuckup in the feature drop code, the files have a shit
    ton of useless data in them.
    Basically the last item in the list is the feature list and the second last
    is the scores we care about.

'''

if __name__ == '__main__':

    p = ap.ArgumentParser()
    p.add_argument("--final", default = "final.pkl.gz")
    p.add_argument("--flist", default = "final.pkl.gz")
    args = p.parse_args()


    features_pos_scores = {}

    for i in range(200):
      # res = zload("DATA/fi-F-%s-DUAL.pkl.gz" % i)
        res = zload(args.final)
        precisions = res[0]
        recalls = res[1]
        fnames = res[2]
        for j in range(len(fnames)-1):
            f1 = 2 * (precisions[j] * recalls[j]) / (precisions[j] + recalls[j])
            f1_next = 2 * (precisions[j+1] * recalls[j+1]) / (precisions[j+1] + recalls[j+1])
            if fnames[j][1] in features_pos_scores:
                features_pos_scores[fnames[j][1]] += (f1 - f1_next)
            else:
                features_pos_scores[fnames[j][1]] = (f1 - f1_next)
        j = len(fnames)-1
        f1 = 2 * (precisions[j] * recalls[j]) / (precisions[j] + recalls[j])
        f1_next = 0
        if fnames[j][1] in features_pos_scores:
            features_pos_scores[fnames[j][1]] += (f1 - f1_next)
        else:
            features_pos_scores[fnames[j][1]] = (f1 - f1_next)
    


    flist = []
    for f, s in features_pos_scores.items():
        flist.append((s, f))

    flist.sort()
    print flist
    zdump(flist, args.flist)


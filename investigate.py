#! /usr/bin/env python

import os, sys, pickle, random, glob, gzip
import argparse as ap

from misc import *


p = ap.ArgumentParser()
p.add_argument("number", type = int)
args = p.parse_args()


print "Dumping res for number %s:" % args.number

number = args.number

order = zload("flist-%s.pkl.gz" % number)
ablation = zload("final-%s.pkl.gz" % (number + 1))


for i in range(59, 75):
    print "%0.3f" % ablation[0][i], order[i][1]

#! /usr/bin/python -u

import os, sys, pickle, random, glob, gzip, math
import argparse as ap
from scipy.stats.stats import pearsonr
from analysis_tools import *
from misc import *
import igraph
import random




def get_centralities(g, target_uid):
    # Compute centralities
    t_vertex = g.vs.select(name = str(int(target_uid)))
    g.vs['eigenvector_centrality'] = g.eigenvector_centrality()
    eigenvector = g.vs['eigenvector_centrality'][t_vertex[0].index]
    AEC = sum(g.vs['eigenvector_centrality']) / len(g.vs)

    g.vs['pagerank_centrality'] = g.pagerank()
    pagerank = g.vs['pagerank_centrality'][t_vertex[0].index]
    APR = sum(g.vs['pagerank_centrality']) / len(g.vs)

    g.vs['betweeness_centrality'] = g.betweenness(g.vs)
    betweenness = g.vs['betweeness_centrality'][t_vertex[0].index]
    ABET = sum(g.vs['betweeness_centrality']) / len(g.vs)

    g.vs['closeness'] = g.closeness(vertices = g.vs, mode = igraph.IN, weights = [e['weight'] for e in g.es], normalized = False)
    closeness = g.vs['closeness'][t_vertex[0].index]
    ACLOSE = sum(g.vs['closeness']) / len(g.vs)

    g.vs['eccentricity'] = g.eccentricity()
    eccentricity = g.vs['eccentricity'][t_vertex[0].index]
    AECC = sum(g.vs['eccentricity']) / len(g.vs)

    g.vs['hub'] = g.hub_score()
    hub = g.vs['hub'][t_vertex[0].index]
    AHUB = sum(g.vs['hub']) / len(g.vs)


    g.vs['degree'] = g.degree(g.vs)
    # Get centralities for target uid
    degree = t_vertex[0].degree()
    ADEG = sum(g.vs['degree']) / len(g.vs)


    AD = g.assortativity_degree(directed=False)
    AAS = g.authority_score(weights=[e['weight'] for e in g.es])
    NAS = AAS[t_vertex[0].index]
    AAS = sum(AAS) / len(AAS)

    APL = g.average_path_length(directed=False)

    CN = g.clique_number()

    g.vs['CORE'] = g.coreness()
    CORE = g.vs['CORE'][t_vertex[0].index]
    ACORE = sum(g.vs['CORE']) / len(g.vs)

    density = g.density()
    diameter = g.diameter(directed=False)

    ecount = len(g.es)
    vcount = len(g.vs)

    g.vs['strength'] = g.strength(g.vs, mode=igraph.ALL, loops=True, weights = [e['weight'] for e in g.es])
    strength = g.vs['strength'][t_vertex[0].index]

    out = [degree, eigenvector, closeness, eccentricity, betweenness, pagerank,     hub, AD, AAS, NAS, APL, CN, CORE, density, diameter, ecount, vcount, AEC, APR, ABET, ACLOSE, AECC, AHUB, ADEG, ACORE]

  # out = [hub]
  # print out
  # out = [random.random(), random.random()]

    out_no_nan = []
    for v in out:
        if math.isnan(v):
            v = -1
        out_no_nan.append(v)

    return out_no_nan



def get_feature_names():
    return ["Degree_Centrality", "Eigenvector_Centrality", "Closeness", "Eccentricity", "Betweenness", "Pagerank_Centrality", "Hub", "assortativity_degree", "average_authority_score",  "authority_score", "average_path_length", "clique_number", "coreness", "density", "diameter", "edge_count", "vertex_count", "average_eigenvector_centrality", "average_pagerank_centrality", "average_betweeness_centrality", "average_closeness", "average_eccentricity", "average_hub", "average_degree", "average_coreness"]




def target_graphml_filenames(t):
    full_graphml = t
    before_graphml = t[:-12] + "before.graphml"
    after_graphml = t[:-12] + "after.graphml"
    return full_graphml, before_graphml, after_graphml



def get_all_centralities(target_uid, filenames):
    out = []
    for filename in filenames:
        g = igraph.Graph.Read_GraphML(filename)
        c = get_centralities(g, target_uid)
        out.extend(c)
    return list(out)



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


def load_labels():
    gen = ranges_gen()
    train, test = gen.next()
    labels = {}
    for p in train:
        X, part_labels = load_full_comments(p)
        labels[p] = part_labels
    for p in test:
        X, part_labels = load_full_comments(p)
        labels[p] = part_labels
    return labels

if __name__ == '__main__':


    res_train = zload("features/centralities-0-train-noscale-multi.pkl.gz")
    res_test = zload("features/centralities-0-test-noscale-multi.pkl.gz")

    centralities = []
    for r in res_train:
        fns = r[0]
        p, i = fns[0].split("/")[1:3]
        p, i = int(p), int(i)
        centralities.append(r[1])
    for r in res_test:
        fns = r[0]
        p, i = fns[0].split("/")[1:3]
        p, i = int(p), int(i)
        centralities.append(r[1])

    ordered_labels = load_labels()
    print len(ordered_labels)

    labels = []
    for r in res_train:
        fns = r[0]
        p, i = fns[0].split("/")[1:3]
        p, i = int(p), int(i)
        print p, i
        labels.append(ordered_labels[p][i])
    for r in res_test:
        fns = r[0]
        p, i = fns[0].split("/")[1:3]
        p, i = int(p), int(i)
        print p, i
        labels.append(ordered_labels[p][i])


    c_names = get_feature_names()


    j = 0
    for i in range(len(centralities[0])):
        ci = [c[i] for c in centralities]
        correlation, p = pearsonr(ci, labels)
        print i, c_names[j], labels[i], len(ci), "%0.2f" % correlation, p
        j += 1
        if j >= len(c_names):
            j = 0
        


# Katz
# Closeness
# Excentricity






#! /usr/bin/python -u

import os, sys, pickle, random, glob, gzip, math, time
import argparse as ap
from scipy.stats.stats import pearsonr
from analysis_tools import *
from misc import *
import igraph
import random


def avg(l):
    return sum(l) / float(len(l))




class FeaturesExtractor:

    def __init__(self, g, target_uid, stopwatch = None):
        self.g = g
        self.nV = len(g.vs)
        self.nE = len(g.es)
        self.target_uid = target_uid
        self.t_vertex = g.vs.select(name = str(int(target_uid)))
        self.t_vertex_i = self.t_vertex[0].index
        self.weigths = [e['weight'] for e in g.es]

        self.skip_list = None
        self.f_dict = None



    def strip_nan(self, out):
        out_no_nan = []
        for v in out:
            if math.isnan(v):
                v = -1
            out_no_nan.append(v)
        return out_no_nan


    def extract_features_from_graph(self, skip_list):
        f_dict = {}
        


    def extract_eigenvector(self):
        out = {}

        eigen_c_w0_d0 = g.eigenvector_centrality(weigths=None, directed=False)
        eigen_c_w0_d0_a = avg(eigen_c_w0_d0)
        out['eigen_c_w0_d0'] = eigen_c_w0_d0[self.t_vertex_i]
        out['eigen_c_w0_d0_a'] = eigen_c_w0_d0_a

        eigen_c_w0_d1 = g.eigenvector_centrality(weigths=None, directed=True)
        eigen_c_w0_d1_a = avg(eigen_c_w0_d1)
        out['eigen_c_w0_d1'] = eigen_c_w0_d1[self.t_vertex_i]
        out['eigen_c_w0_d1_a'] = eigen_c_w0_d1_a

        eigen_c_w1_d0 = g.eigenvector_centrality(weigths=self.weigths, directed=False)
        eigen_c_w1_d0_a = avg(eigen_c_w1_d0)
        out['eigen_c_w1_d0'] = eigen_c_w1_d0[self.t_vertex_i]
        out['eigen_c_w1_d0_a'] = eigen_c_w1_d0_a

        eigen_c_w1_d1 = g.eigenvector_centrality(weigths=self.weigths, directed=True)
        eigen_c_w1_d1_a = avg(eigen_c_w1_d1)
        out['eigen_c_w1_d1'] = eigen_c_w1_d1[self.t_vertex_i]
        out['eigen_c_w1_d1_a'] = eigen_c_w1_d1_a

        return out


    def extract_pagerank(self):
        out = {}

        pr_c_w0_d0 = g.pagerank(weigths=None, directed=False)
        pr_c_w0_d0_a = avg(pr_c_w0_d0)
        out['pr_c_w0_d0'] = pr_c_w0_d0[self.t_vertex_i]
        out['pr_c_w0_d0_a'] = pr_c_w0_d0_a

        pr_c_w0_d1 = g.pagerank(weigths=None, directed=True)
        pr_c_w0_d1_a = avg(pr_c_w0_d1)
        out['pr_c_w0_d1'] = pr_c_w0_d1[self.t_vertex_i]
        out['pr_c_w0_d1_a'] = pr_c_w0_d1_a

        pr_c_w1_d0 = g.pagerank(weigths=self.weigths, directed=False)
        pr_c_w1_d0_a = avg(pr_c_w1_d0)
        out['pr_c_w1_d0'] = pr_c_w1_d0[self.t_vertex_i]
        out['pr_c_w1_d0_a'] = pr_c_w1_d0_a

        pr_c_w1_d1 = g.pagerank(weigths=self.weigths, directed=True)
        pr_c_w1_d1_a = avg(pr_c_w1_d1)
        out['pr_c_w1_d1'] = pr_c_w1_d1[self.t_vertex_i]
        out['pr_c_w1_d1_a'] = pr_c_w1_d1_a

        return out


    def extract_betweeness(self):
        out = {}

        be_c_w0_d0 = g.betweenness(weigths=None, directed=False)
        be_c_w0_d0_a = avg(be_c_w0_d0)
        out['be_c_w0_d0'] = be_c_w0_d0[self.t_vertex_i]
        out['be_c_w0_d0_a'] = be_c_w0_d0_a

        be_c_w0_d1 = g.betweenness(weigths=None, directed=True)
        be_c_w0_d1_a = avg(be_c_w0_d1)
        out['be_c_w0_d1'] = be_c_w0_d1[self.t_vertex_i]
        out['be_c_w0_d1_a'] = be_c_w0_d1_a

        be_c_w1_d0 = g.betweenness(weigths=self.weigths, directed=False)
        be_c_w1_d0_a = avg(be_c_w1_d0)
        out['be_c_w1_d0'] = be_c_w1_d0[self.t_vertex_i]
        out['be_c_w1_d0_a'] = be_c_w1_d0_a

        be_c_w1_d1 = g.betweenness(weigths=self.weigths, directed=True)
        be_c_w1_d1_a = avg(be_c_w1_d1)
        out['be_c_w1_d1'] = be_c_w1_d1[self.t_vertex_i]
        out['be_c_w1_d1_a'] = be_c_w1_d1_a

        return out


    def extract_closeness(self):
        out = {}

        in_clo_w0 = g.closeness(mode=igraph.IN, weights=None)
        in_clo_w0_a = avg(in_clo_w0)
        in_clo_w1 = g.closeness(mode=igraph.IN, weights=self.weights)
        in_clo_w1_a = avg(in_clo_w1)
        out['in_clo_w0'] = in_clo_w0[self.t_vertex_i]
        out['in_clo_w0_a'] = in_clo_w0_a
        out['in_clo_w1'] = in_clo_w1[self.t_vertex_i]
        out['in_clo_w1_a'] = in_clo_w1_a

        out_clo_w0 = g.closeness(mode=igraph.OUT, weights=None)
        out_clo_w0_a = avg(out_clo_w0)
        out_clo_w1 = g.closeness(mode=igraph.OUT, weights=self.weights)
        out_clo_w1_a = avg(out_clo_w1)
        out['out_clo_w0'] = out_clo_w0[self.t_vertex_i]
        out['out_clo_w0_a'] = out_clo_w0_a
        out['out_clo_w1'] = out_clo_w1[self.t_vertex_i]
        out['out_clo_w1_a'] = out_clo_w1_a

        all_clo_w0 = g.closeness(mode=igraph.ALL, weights=None)
        all_clo_w0_a = avg(all_clo_w0)
        all_clo_w1 = g.closeness(mode=igraph.ALL, weights=self.weights)
        all_clo_w1_a = avg(all_clo_w1)
        out['all_clo_w0'] = all_clo_w0[self.t_vertex_i]
        out['all_clo_w0_a'] = all_clo_w0_a
        out['all_clo_w1'] = all_clo_w1[self.t_vertex_i]
        out['all_clo_w1_a'] = all_clo_w1_a        

        return out


    def extract_eccentricity(self):
        out = {}

        in_ecc = g.eccentricity(mode=igraph.IN)
        in_ecc_a = avg(in_ecc)
        out['in_ecc'] = in_ecc[self.t_vertex_i]
        out['in_ecc_a'] = in_ecc_a

        out_ecc = g.eccentricity(mode=igraph.OUT)
        out_ecc_a = avg(out_ecc)
        out['out_ecc'] = out_ecc[self.t_vertex_i]
        out['out_ecc_a'] = out_ecc_a

        all_ecc = g.eccentricity(mode=igraph.ALL)
        all_ecc_a = avg(all_ecc)
        out['all_ecc'] = all_ecc[self.t_vertex_i]
        out['all_ecc_a'] = all_ecc_a

        return out


    def extract_hub(self):
        out = {}

        hub_w0 = g.hub_score(weights = None)
        hub_w0_a = avg(hub_w0)
        out['hub_w0'] = hub_w0[self.t_vertex_i]
        out['hub_w0_a'] = hub_w0_a

        hub_w1 = g.hub_score(weights = self.weights)
        hub_w1_a = avg(hub_w1)
        out['hub_w1'] = hub_w1[self.t_vertex_i]
        out['hub_w1_a'] = hub_w1_a

        return out


    def extract_degree(self):
        out = {}

        in_deg = g.degree(mode=igraph.IN)
        in_deg_a = avg(in_deg)
        out['in_deg'] = in_deg[self.t_vertex_i]
        out['in_deg_a'] = in_deg_a

        out_deg = g.degree(mode=igraph.OUT)
        out_deg_a = avg(out_deg)
        out['out_deg'] = out_deg[self.t_vertex_i]
        out['out_deg_a'] = out_deg_a

        all_deg = g.degree(mode=igraph.ALL)
        all_deg_a = avg(all_deg)
        out['all_deg'] = all_deg[self.t_vertex_i]
        out['all_deg_a'] = all_deg_a

        return out


    def extract_assortativity_degree(self):
        out = {}

        ad_d0 = g.assortativity_degree(directed=False)
        out['ad_d0'] = ad_d0
        ad_d1 = g.assortativity_degree(directed=True)
        out['ad_d1'] = ad_d1

        return out


    def extract_authority_score(self):
        out = {}

        as_w0 = authority_score(weigths=None)
        as_w0_a = avg(as_w0)
        out['as_w0'] = as_w0[self.t_vertex_i]
        out['as_w0_a'] = as_w0_a

        as_w1 = authority_score(weights=self.weights)
        as_w1_a = avg(as_w1)
        out['as_w1'] = as_w1[self.t_vertex_i]
        out['as_w1_a'] = as_w1_a

        return out


    def extract_average_path_length(self):
        out = {}

        apl_d0 = g.average_path_length(directed=False)
        out['apl_d0'] = apl_d0

        apl_d1 = g.average_path_length(directed=True)
        out['apl_d1'] = apl_d1

        return out


    def extract_clique_number(self):
        out = {}

        out['clique_number'] = g.clique_number()

        return out


    def extract_coreness(self):
        out = {}

        in_core = g.coreness(mode=igraph.IN)
        in_core_a = avg(in_core)
        out['in_core'] = in_core
        out['in_core_a'] = in_core_a

        out_core = g.coreness(mode=igraph.OUT)
        out_core_a = avg(out_core)
        out['out_core'] = out_core
        out['out_core_a'] = out_core_a

        all_core = g.coreness(mode=igraph.ALL)
        all_core_a = avg(all_core)
        out['all_core'] = all_core
        out['all_core_a'] = all_core_a

        return out


    def extract_density(self):
        out = {}

        out['density'] = g.density()

        return out


    def extract_diameter(self):
        out = {}

        dia_w0_d0 = g.diameter(weigths=None, directed=False)
        dia_w0_d1 = g.diameter(weigths=None, directed=True)
        dia_w1_d0 = g.diameter(weigths=self.weights, directed=False)
        dia_w1_d1 = g.diameter(weigths=self.weights, directed=True)
        out['dia_w0_d0'] = dia_w0_d0
        out['dia_w0_d1'] = dia_w0_d1
        out['dia_w1_d0'] = dia_w1_d0
        out['dia_w1_d1'] = dia_w1_d1

        return out


    def extract_edge_count(self):
        out = {}
        out['edge_count'] = len(g.es)
        return out


    def extract_vertice_count(self):
        out = {}
        out['vertice_count'] = len(g.vs)
        return out


    def extract_strength(self):
        out = {}

        in_str_w0 = g.strength(mode=igraph.IN, weights=None)
        in_str_w0_a = avg(in_str_w0)
        in_str_w1 = g.strength(mode=igraph.IN, weights=self.weights)
        in_str_w1_a = avg(in_str_w1)
        out['in_str_w0'] = in_str_w0[self.t_vertex_i]
        out['in_str_w0_a'] = in_str_w0_a
        out['in_str_w1'] = in_str_w1[self.t_vertex_i]
        out['in_str_w1_a'] = in_str_w1_a

        out_str_w0 = g.strength(mode=igraph.OUT, weights=None)
        out_str_w0_a = avg(out_str_w0)
        out_str_w1 = g.strength(mode=igraph.OUT, weights=self.weights)
        out_str_w1_a = avg(out_str_w1)
        out['out_str_w0'] = out_str_w0[self.t_vertex_i]
        out['out_str_w0_a'] = out_str_w0_a
        out['out_str_w1'] = out_str_w1[self.t_vertex_i]
        out['out_str_w1_a'] = out_str_w1_a

        all_str_w0 = g.strength(mode=igraph.ALL, weights=None)
        all_str_w0_a = avg(all_str_w0)
        all_str_w1 = g.strength(mode=igraph.ALL, weights=self.weights)
        all_str_w1_a = avg(all_str_w1)
        out['all_str_w0'] = all_str_w0[self.t_vertex_i]
        out['all_str_w0_a'] = all_str_w0_a
        out['all_str_w1'] = all_str_w1[self.t_vertex_i]
        out['all_str_w1_a'] = all_str_w1_a        

        return out


    def extract_transitivity(self):
        out = {}

        trans_loc_d0_w0 = g.transitivity_local_undirected(vertices=[self.t_vertex_i], weights=None, mode="zero")
        out['trans_loc_d0_w0'] = trans_loc_d0_w0
        trans_loc_d0_w1 = g.transitivity_local_undirected(vertices=[self.t_vertex_i], weights=self.weigths, mode="zero")
        out['trans_loc_d0_w1'] = trans_loc_d0_w1

        trans_avgloc_d0 = g.transitivity_avglocal_undirected(mode="zero")
        out['trans_avgloc_d0'] = trans_avgloc_d0

        trans_d0 = g.transitivity_undirected(mode="zero")
        out['trans_d0'] = trans_d0

        return out






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






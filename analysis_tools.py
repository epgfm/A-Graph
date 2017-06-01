#! /usr/bin/env python

import json
from misc import *



def load_graph_data(json_file):
    ''' (str) -> dict

    Returns the data used to create the graph as a dictionnary.

    >>> print [key for key in load_graph_data("../0/0/full.json")]
    [u'uid_names_dict', u'a_target_row', u'rows', u'channel', u'label']
    '''
    with open(json_file) as f:
        json_str = f.read()
    return json.loads(json_str)



def get_target_uid(graph_data):
    ''' (dict) -> str
    Gets the uid of the targeted user from graph data

    >>> get_target_uid(load_graph_data("../0/0/full.json"))
    u'30050'
    '''
    target_row_indice = graph_data["a_target_row"]
    target_row = graph_data['rows'][target_row_indice]
    # Build dict of names: uid
    names_uid_dict = {}
    uid_names_dict = graph_data["uid_names_dict"]
    for uid in uid_names_dict:
        ircname = uid_names_dict[uid][0]
        if ircname not in names_uid_dict:
            names_uid_dict[ircname] = uid
    # get ircname from target row
    target_ircname = target_row.split()[1][1:-1] # Strip <>
    return names_uid_dict[target_ircname]




if __name__ == "__main__":
    import doctest
    doctest.testmod()

    





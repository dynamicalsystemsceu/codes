#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from graph_tool.all import *
import collections
import csv


# # 1. Loading data

# # 2 Shuffling timestamps
# - Timestamp column is shuffled, while columns "from" and "to" is kept fixed.

# In[3]:


# This is the function that is used for shuffling
def shuffle_timestamps_edgelist(edgelist,random_state):
    edgelist = edgelist.copy()
    edgelist['t_second'] = edgelist['t_second'].sample(frac=1,random_state=random_state+5000).values
    edgelist.reset_index(inplace=True, drop=True)
    return edgelist





def rewire_shuffle_timestamps_edgelist(edgelist,random_state):
    edgelist = edgelist.copy()
    edgelist['from'] = edgelist['from'].sample(frac=1,random_state=random_state).values
    edgelist['to'] = edgelist['to'].sample(frac=1,random_state=random_state+2000).values
    edgelist['t_second'] = edgelist['t_second'].sample(frac=1,random_state=random_state+3000).values
    edgelist.reset_index(inplace=True, drop=True)
    return edgelist



def remove_sublists(lst):
     return ([list(i) for i in {*[tuple(sorted(i)) for i in lst]}])  
def nodes_unique_pairs(df_edges):
    link_pairs = list(zip(df_edges["from"], df_edges["to"]))
    nodes = pd.unique(list(df_edges['to']) + list(df_edges['from']))
    np.random.shuffle(nodes)
    link_pairs =[[str(pair[0]),str(pair[1])] for pair in link_pairs if len(pair) > 1]
    unique_pairs = []
    for pair in link_pairs:
        if pair[0] >= pair[1]:
            unique_pairs.append(pair)
        else:
            unique_pairs.append([pair[1],pair[0]])
    unique_pairs = remove_sublists(unique_pairs)
    return nodes,unique_pairs 

def self_loops_edgelist(df_edges):
    link_pairs = list(zip(df_edges["from"], df_edges["to"]))
    nodes = pd.unique(list(df_edges['to']) + list(df_edges['from']))
    np.random.shuffle(nodes)
    link_pairs =[[str(pair[0]),str(pair[1])] for pair in link_pairs if len(pair) > 1]
    unique_pairs = []
    for pair in link_pairs:
        if pair[0] == pair[1]:
            unique_pairs.append(pair)
   # unique_pairs = remove_sublists(unique_pairs)
    return unique_pairs 


def create_link_edgelist(edgelist,node_id_1,node_id_2):
    edgelist = edgelist.copy()
    link_edgelist = edgelist[((edgelist['from'].astype(str) == str(node_id_1) )& (edgelist['to'].astype(str) == str(node_id_2)))|((edgelist['from'].astype(str) == str(node_id_2)) & (edgelist['to'].astype(str) == str(node_id_1)))]
    return link_edgelist

# In[9]:


def shuffle_link_edgelist(link_edgelist,random_state):
    link_edgelist = link_edgelist.copy()
    link_edgelist['t_second'] = link_edgelist['t_second'].sample(frac=1,random_state=random_state+6000).values
    return link_edgelist


# Links shuffles


def shuffle_timestamps_on_links(df_edges,random_state):
    df_edges = df_edges.copy()
    nodes,unique_pairs = nodes_unique_pairs(df_edges)
    all_links_list_df = []
    for pair in unique_pairs:
        link_edgelist = create_link_edgelist(df_edges,pair[0],pair[1])
        #df_edges = df_edges.drop(link_edgelist.index)
        link_edgelist_shuffled = shuffle_link_edgelist(link_edgelist,random_state)
        all_links_list_df.append(link_edgelist)
    link_edgelist_shuffled = pd.concat(all_links_list_df)
    link_edgelist_shuffled.reset_index(inplace=True, drop=True)
    return  link_edgelist_shuffled

# Node shuffles


def create_node_edgelist(edgelist,node_id):
    edgelist = edgelist.copy()
    node_edgelist = edgelist[(edgelist['from'].astype(str) == str(node_id)) | (edgelist['to'].astype(str) == str(node_id)) ]
    return node_edgelist.astype(str)


def shuffle_node_edgelist(node_edgelist,random_state):
    node_edgelist = node_edgelist.copy()
    node_edgelist['t_second'] =node_edgelist['t_second'].sample(frac=1,random_state=random_state+7000).values
    return node_edgelist



#Function for the entire edgelist
def shuffle_timestamps_at_nodes(df_edges,random_state):
    df_edges = df_edges.copy()
    nodes,unique_pairs = nodes_unique_pairs(df_edges)
    nodes_shuffled_list_df = []
    for node in nodes:
        node_edgelist = create_node_edgelist(df_edges,node)
        df_edges = df_edges.drop(node_edgelist.index)
        node_edgelist_shuffled = shuffle_node_edgelist(node_edgelist,random_state)
        nodes_shuffled_list_df.append(node_edgelist_shuffled)
    node_edgelist_shuffled = pd.concat(nodes_shuffled_list_df)
    node_edgelist_shuffled.reset_index(inplace=True, drop=True)
    return node_edgelist_shuffled





# # 7 Rewiring sources and fixing timestamps at targets
# - OUT degree constant with rewiring







def rewire_shuffle_node_edgelist(node_edgelist,node_id,random_state):
    node_edgelist = node_edgelist.astype(str).copy()
    node_id = str(node_id)
    count_to_central = 0
    count_from_central = 0
    to_list = list(node_edgelist ['to'])
    from_list = list(node_edgelist ['from'])
    count_to_central = to_list.count(node_id)
    count_from_central = from_list.count(node_id)
    if len(from_list)>0 and len(to_list)>0 and (len(from_list) != count_from_central and len(to_list) != count_to_central):
        local_nodes_no_central_node = list(pd.unique(to_list + from_list))
        local_nodes_no_central_node.remove(node_id)
        np.random.seed(random_state)
        if count_from_central > 0 and count_to_central>0:
            to_local_list = np.random.choice(local_nodes_no_central_node,size=abs(count_to_central-len(to_list)),replace=True)
            from_central_list = [node_id for i in range(0,count_from_central)]
            from_local_list =  np.random.choice(local_nodes_no_central_node,size=abs(count_from_central-len(from_list)),replace=True)
            to_central_list = [node_id for i in range(0,count_to_central)]
            node_edgelist['from'] = np.append(from_central_list,from_local_list)
            node_edgelist['to'] = np.append(to_local_list,to_central_list)
        if count_from_central > 0 and count_to_central == 0:
            to_local_list = np.random.choice(local_nodes_no_central_node,size==abs(count_from_central-len(from_list)),replace=True)
            from_central_list = [node_id for i in range(0,count_from_central)]
            node_edgelist['from'] = from_central_list 
            node_edgelist['to'] = to_local_list
        if count_from_central == 0 and count_to_central > 0:
            from_local_list =  np.random.choice(local_nodes_no_central_node,size=abs(count_to_central-len(to_list)),replace=True)
            to_central_list = [node_id for i in range(0,count_to_central)]
            node_edgelist['from'] = from_local_list 
            node_edgelist['to'] = to_central_list

    node_edgelist['t_second'] =node_edgelist['t_second'].sample(frac=1,random_state=random_state+13000).values
    node_edgelist.reset_index(inplace=True, drop=True)
    return node_edgelist



def rewire_shuffle_timestamps_at_nodes(df_edges,random_state):
    df_edges = df_edges.copy()
    nodes,unique_pairs = nodes_unique_pairs(df_edges)
    nodes_shuffled_list_df = []
    for node in nodes:
        node_edgelist = create_node_edgelist(df_edges,node)
        df_edges = df_edges.drop(node_edgelist.index)
        node_edgelist_shuffled = rewire_shuffle_node_edgelist(node_edgelist,node,random_state)
        nodes_shuffled_list_df.append(node_edgelist_shuffled )
    node_edgelist_shuffled = pd.concat(nodes_shuffled_list_df)
    node_edgelist_shuffled.reset_index(inplace=True, drop=True)
    return node_edgelist_shuffled





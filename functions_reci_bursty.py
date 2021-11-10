from graph_tool.all import *
import pickle
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

import collections
import csv

# RECIPROCAL EVENTS
#____________________


# Function which take in input the graphs (g_D,g) and return (g_D) with properties of the nodes: 
    # - 1. number of reciprocal events: 'n_rec'
    # - 2. probability of reciprocal events: 'proba_rec_event'
    # - 3. probability of reciprocal links : 'proba_rec_link'

def rec_nodes(g_D,g):
    prop =[g_D.ep[p] for p in dict(g_D.edge_properties).keys()]
    
    
    n_rec_event = g.new_vertex_property("double")
    g.vp.n_rec_event= n_rec_event
    
    n_rec_link = g.new_vertex_property("double")
    g.vp.n_rec_link= n_rec_link
      
    proba_rec_event = g.new_vertex_property("double") 
    g.vp.proba_rec_event= proba_rec_event
    
    proba_rec_link = g.new_vertex_property("double")     
    g.vp.proba_rec_link= proba_rec_link
    
    
    #---
    n_tot_events = g.new_vertex_property("double") 
    g.vp.n_tot_events= n_tot_events
    
    
    for node in g_D.vertices():
        counter_rec_event = 0
        counter_rec_link = 0
        
        events = g_D.get_all_edges(node,eprops = prop)
        neighbors = g.get_all_neighbors(node)
        
        
        #---
        g.vp.n_tot_events[node] = len(events)
        
        for neighbor in neighbors:
            rows, cols = np.where(events[:,:2] == neighbor)
            events_node_nei = sorted(events[rows], key = lambda x: x[5])
            
            binary_rec = [0 if events_node_nei[k][0]==events_node_nei[k+1][0] 
                          else 1 for k in range(len(events_node_nei)-1)].count(1)
            
            # counting n of rec events (on g_D)
            counter_rec_event += binary_rec
            
            
            # countinf n of rec links (on g)
            if binary_rec !=0: 
                counter_rec_link +=1
            
        g.vp.n_rec_event[node] = counter_rec_event
        g.vp.n_rec_link[node] = counter_rec_link
        
        
        
        if len(events)==0:
            g.vp.proba_rec_event[node] = 0
            g.vp.proba_rec_link[node] = 0
        else:
            #print('nn', len(neighbors))
            #print('e', len(events)) 
            g.vp.proba_rec_event[node] = counter_rec_event / (len(events))
            g.vp.proba_rec_link[node] = counter_rec_link / (len(neighbors))
               
    return(g)


# BURSTINESS
#____________

def burstiness(distri_intertimes):
    '''
    range: from -1 (deterministic) to +1 (super bursty)
    '''
    
    mean = np.mean(distri_intertimes)
    std = np.std(distri_intertimes)
    return((std-mean)/(std+mean))



# Reciprocal events
#--------------------

# Function which takes as an input g_D and return it with the nodes property:
    # - 1. burtiness of the reciprocal 'burst_rec'
    # - 2. intertime dist of the reciprocal event 'intertime_rec'
    
def burst_rec_nodes(g_D,g):
    
    prop =[g_D.ep[p] for p in dict(g_D.edge_properties).keys()]
    
    burst_rec = g.new_vertex_property("double") 
    intertime_rec = g.new_vertex_property("vector<double>") 
    
    g.vp.burst_rec= burst_rec
    g.vp.intertime_rec= intertime_rec
    
    for node in g_D.vertices():
        
        list_burstiness_rec = []
        node_intertimes_rec = []
        
        events = g_D.get_all_edges(node,eprops = prop)
    
        for neighbor in g.get_all_neighbors(node):
            rows, cols = np.where(events[:,:2] == neighbor)
            events_node_nei = sorted(events[rows], key = lambda x: x[5])
            
            intertime_rec = [events_node_nei[k+1][5]-events_node_nei[k][5]
                             for k in range(len(events_node_nei)-1) if
                             events_node_nei[k][0]!=events_node_nei[k+1][0]]
            
            node_intertimes_rec.extend(intertime_rec) 
            list_burstiness_rec += intertime_rec
        
        #print(node_intertimes_rec)
        # Crating properties 
        g.vp.burst_rec[node] = burstiness(list_burstiness_rec)
        g.vp.intertime_rec[node] = np.array(node_intertimes_rec)
        
        
    return(g)


# NON- Reciprocal events
#-----------------------

# Function which takes as an input g_D and return it with the nodes property: 
    # - 1. burstiness of the non-reciprocal event: 'burst_no_rec'
    # - 2. intertime dist of the non-reciprocal event: 'intertime_no_rec'

def burst_no_rec_nodes(g_D,g):
    
    prop =[g_D.ep[p] for p in dict(g_D.edge_properties).keys()]
    
    burst_no_rec = g.new_vertex_property("double") 
    intertime_no_rec = g.new_vertex_property("vector<double>") 
    
    g.vp.burst_no_rec= burst_no_rec
    g.vp.intertime_no_rec= intertime_no_rec
    
    for node in g_D.vertices():
        
        list_burstiness_no_rec = []
        node_intertimes_no_rec = []
        
        events = g_D.get_all_edges(node,eprops = prop)
        
        for neighbor in g.get_all_neighbors(node):
        
            rows, cols = np.where(events[:,:2] == neighbor)
            events_node_nei = sorted(events[rows], key = lambda x: x[5])
            
            intertime_no_rec = [events_node_nei[k+1][5]-events_node_nei[k][5] 
                               for k in range(len(events_node_nei)-1) 
                                 if events_node_nei[k][0]==events_node_nei[k+1][0]]
            
            list_burstiness_no_rec += intertime_no_rec
            node_intertimes_no_rec.extend(intertime_no_rec)
            
            
        g.vp.burst_no_rec[node] = burstiness(list_burstiness_no_rec)
        g.vp.intertime_no_rec[node] = np.array(node_intertimes_no_rec)
        
    return(g)


# ALL events
#-----------------------

# Function which takes as an input g_D and return it with the nodes property: 
    # - 1. burstiness of the all event: 'burst'
    # - 2. intertime dist of all event: 'intertime'

def burst_nodes(g_D,g):
    
    prop =[g_D.ep[p] for p in dict(g_D.edge_properties).keys()]
    
    burst = g.new_vertex_property("double") 
    intertime = g.new_vertex_property("vector<double>") 
    
    g.vp.burst= burst
    g.vp.intertime= intertime
    
    for node in g_D.vertices():
        list_burstiness= []
        node_intertimes= []
        
        events = g_D.get_all_edges(node,eprops = prop)
        
        for neighbor in g.get_all_neighbors(node):
        
            rows, cols = np.where(events[:,:2] == neighbor)
            events_node_nei = sorted(events[rows], key = lambda x: x[5])
            
            intertime = [events_node_nei[k+1][5]-events_node_nei[k][5] 
                         for k in range(len(events_node_nei)-1)]
            
            list_burstiness += intertime
            node_intertimes.extend(intertime)            
            
        g.vp.burst[node] = burstiness(list_burstiness)
        
        g.vp.intertime[node] = np.array(node_intertimes)
        
    return(g)


#---------------------------------------------
# calculating edge properties
#---------------------------------------------
def compute_link_prop(g,g_D):
    
    '''
    Note the edge properties will be saved  on g!!! 
        ie. on the aggregate static network
    
    In this function we compute: 
    - Balance at edge level: Given e edge from i to j.
        be = max(Ni, Nj)/(Ni+Nj)
   
    - P_Erec at edge level: Probability of having a reciprocal event for a given sequence of events be node ij
        p_Erec = n_rec/(ni+nj-1)
        
    - Intertime
    
    - Burstiness at edge level
    
    '''
    
    # 1. Creating edge (link) property 
    balance = g.new_edge_property("double") 
    g.ep.balance= balance
    
    p_Erec = g.new_edge_property("double") # prop of reciprocity at link level 
    g.ep.p_Erec= p_Erec
    
    burts = g.new_edge_property("double") 
    g.ep.burts= burts
    
    n_events= g.new_edge_property("double") 
    g.ep.n_events= n_events
    
    intertime = g.new_edge_property("vector<double>") 
    g.ep.intertime= intertime
    
    
    ite= 1
    N=str(g.num_edges())
    
    for e in g.edges(): # for every edges in the g graph
        sys.stdout.write('\r' +'  edges n: '+ str(ite)+'/'+N )

        i = e.source() # node i
        j = e.target() # node j
        
        prop =[g_D.ep[p] for p in dict(g_D.edge_properties).keys()]
        all_edges_i = np.array([e for e in g_D.get_all_edges(i,eprops=prop) if e[1] ==j])
        all_edges_j = np.array([e for e in g_D.get_all_edges(j,eprops=prop) if e[1] ==i])
        
        ni = len(all_edges_i) 
        nj = len(all_edges_j)

        # -  Computing rec and non_rec lists
        # - At least one reciprocal bw ij 
        if ni>0 and nj>0:
            list_events= np.concatenate((all_edges_i, all_edges_j))
            list_events = sorted(list_events, key = lambda x: x[5])
            
            # Counting number of reciprocal 
            binary_reciprocal = ''.join(['0' if list_events[k+1][1]==list_events[k][1] else '1' for k in range(len(list_events)-1)])
            n_rec = binary_reciprocal.count('1')
        
            
        # - Non Reciprocal events bw ij ( at all: no event is reciprocal bw there two)
        elif (ni>0 and nj==0) or (ni==0 and nj>0):            
            if ni>0 and nj==0:
                list_events= np.copy(all_edges_i)  
            elif ni==0 and nj>0:
                list_events= np.copy(all_edges_j)
            n_rec = 0 
        
        
        # -  Intertimes of the link ij
        intertimes = [list_events[k+1][5]-list_events[k][5]  for k in range(len(list_events)-1)]
        
        # -  Balance
        be = max(ni,nj)/(ni+nj)
        g.ep.balance[e] = be
        
        # -  P-rec at link level:
        if (ni+nj-1) == 0: 
            p_Erec = np.NaN
        else:
            p_Erec = n_rec/(ni+nj-1)
        g.ep.p_Erec[e] = p_Erec
        
        # - Intertime 
        g.ep.intertime[e] = np.array(intertimes)
        
        # - Burstiness
        burts = burstiness(intertimes)
        g.ep.burts[e] = burts
        
        # - Number of events between two nodes
        g.ep.n_events[e] = len(list_events)
        
        ite +=1
    return g





def measures(df_edges_calls,XX):

    df_edges_calls_shuffled = df_edges_calls

    g_calls_shuffled,g_D_calls_shuffled = to_graph(df_edges_calls_shuffled,'from','to','t_second','t_minutes','t_hours','t_days')
    REC_shuffled,REC_no_reciprocity_shuffled = compute_rec(g_calls_shuffled,g_D_calls_shuffled)
    N_ev_rec_shuffled, N_reciprocity_shuffled = distribution_rec_interevent(REC_shuffled,g_calls_shuffled)

    node_list_1,node_list_2 = run_dist_node_intertime(g_D_calls_shuffled)
    L_sec_nodes_shuffled = node_list_1[0]

    edge_list_1 = run_dist_edges_intertime(REC_no_reciprocity_shuffled)
    L_sec_edges_shuffled = edge_list_1[0]

    results = {}
    results['Method'] = XX
    # PROBA TO HAVE A RECIPROCAL LINK
    results['P(l_rec)'] = proba_reciprocal_link(REC_shuffled,g_D_calls_shuffled,g_calls_shuffled)
    # PROBA TO HAVE A RECIPROCAL EVENT
    results['P(E_rec)'] = proba_reciprocal_event(g_D_calls_shuffled,g_calls_shuffled,N_reciprocity_shuffled)
    # BURSTINESS FROM NODES POINT OF VIEW
    results['B_nodes'] = burstiness(L_sec_nodes_shuffled)
    # BURSTINESS FROM EDGES POINT OF VIEW
    results['B_edges'] = burstiness(L_sec_edges_shuffled)
        
    
    results_df = pd.DataFrame.from_records([results])
    results_df.set_index("Method", inplace = True)
    return results_df 

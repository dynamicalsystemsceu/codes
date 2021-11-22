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

# FUNCTION WHICH CREATES THE AGGREGATED GRAPH: G IS UNDIRECTED AND G_D IS DIRECTED


def graph_filter_func(g,g_D):
    # FILTERING
    #-------------------------------------
    
#     a. (node filtering) Removing nodes with no reciprocal ecents
#     g_filt = GraphView(g, vfilt=lambda v: g.vp.n_rec_event[v] > 0.0)   

    
    # OLDD OKK ??? 
    #Filtering in and out degree (keep only nodes with degree >=1)
    g_D_filt = GraphView(g_D, vfilt=lambda v: (v.out_degree()>=1)&(v.in_degree()>=1))
    g_filt = GraphView(g, vfilt=lambda v: (v in g_D_filt.vertices())==True)

    # b. (edge filtering) Removing unique edges bw two nodes (ie. if only one event bw two nodes)
    g_filt = GraphView(g_filt, efilt=lambda e: g_filt.ep.n_events[e] > 1.0)
    return g_filt
     


def to_graph(df_edges,from_ ,to,sec,minutes, hours, days):
    
    # Edge list
    elist =np.array(df_edges)#
    #np.array(list(zip(df_edges[from_],df_edges[to],df_edges[sec],df_edges[minutes],df_edges[hours],df_edges[days])))

    # Building the UNdirected network
    g = Graph(directed=False)
    
    #ts_sec = g.new_ep("double")
    #ts_min = g.new_ep("double")
    #ts_h = g.new_ep("double")
    #ts_days = g.new_ep("double")
    vlabel = g.add_edge_list(elist, hashed=True)#, eprops=[ts_sec,ts_min,ts_h,ts_days])

    graph_tool.stats.remove_self_loops(g)  #removing self loops 
    graph_tool.stats.remove_parallel_edges(g) #and multiple edge
    
    #g.ep["ts_sec"] = ts_sec
    #g.ep["ts_min"] = ts_min
    #g.ep["ts_h"] = ts_h
    #g.ep["ts_days"] = ts_days
    #g.vp["label"] = vlabel
    
    ############- Directed 
    g_D = Graph(directed=True)
    
    ts_sec = g_D.new_ep("double")
    ts_min = g_D.new_ep("double")
    ts_h = g_D.new_ep("double")
    ts_days = g_D.new_ep("double")

    vlabel = g_D.add_edge_list(elist, hashed=True, eprops=[ts_sec,ts_min,ts_h,ts_days])
    
    graph_tool.stats.remove_self_loops(g_D)  #removing self loops 
    
    g_D.ep["ts_sec"] = ts_sec
    g_D.ep["ts_min"] = ts_min
    g_D.ep["ts_h"] = ts_h
    g_D.ep["ts_days"] = ts_days
    g_D.vp["label"] = vlabel

    return(g,g_D)




# RECIPROCAL EVENTS
#____________________


# Function which take in input the graphs (g_D,g) and return (g_D) with properties of the nodes: 
    # - 1. number of reciprocal events: 'n_rec'
    # - 2. probability of reciprocal events: 'proba_rec_event'
    # - 3. probability of reciprocal links : 'proba_rec_link'

def rec_nodes(g_D,g):
    properties = list(dict(g_D.edge_properties).keys())
    time_index = properties.index('ts_sec')+2
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
            events_node_nei = sorted(events[rows], key = lambda x: x[time_index])
            
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
    if len(distri_intertimes)>0:
        mean = np.mean(np.array(distri_intertimes))
        std = np.std(np.array(distri_intertimes))
    else:
        mean=np.nan
        std=np.nan
    return((std-mean)/(std+mean))



# Reciprocal events
#--------------------

# Function which takes as an input g_D and return it with the nodes property:
    # - 1. burtiness of the reciprocal 'burst_rec'
    # - 2. intertime dist of the reciprocal event 'intertime_rec'
    
def burst_rec_nodes(g_D,g):
    properties = list(dict(g_D.edge_properties).keys())
    time_index = properties.index('ts_sec')+2
    
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
            events_node_nei = sorted(events[rows], key = lambda x: x[time_index])
            
            intertime_rec = [events_node_nei[k+1][time_index]-events_node_nei[k][time_index]
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
    properties = list(dict(g_D.edge_properties).keys())
    time_index = properties.index('ts_sec')+2    
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
            events_node_nei = sorted(events[rows], key = lambda x: x[time_index])
            
            intertime_no_rec = [events_node_nei[k+1][time_index]-events_node_nei[k][time_index] 
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
    properties = list(dict(g_D.edge_properties).keys())
    time_index = properties.index('ts_sec')+2    
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
            events_node_nei = sorted(events[rows], key = lambda x: x[time_index])
#             print(events[rows],'\n lalalalallala'
            intertime = [events_node_nei[k+1][time_index]-events_node_nei[k][time_index] 
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
    properties = list(dict(g_D.edge_properties).keys())
    time_index = properties.index('ts_sec')+2    
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
            list_events = sorted(list_events, key = lambda x: x[time_index])
            
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
        intertimes = [list_events[k+1][time_index]-list_events[k][time_index]  for k in range(len(list_events)-1)]
        
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





def measures(df_edges,XX):


    g,g_D = to_graph(df_edges,'from','to','t_second','t_minutes','t_hours','t_days')
    
    
    # Do stuff on nodes
    g = rec_nodes(g_D,g)
    g = burst_rec_nodes(g_D,g)
    g = burst_no_rec_nodes(g_D,g)
    g = burst_nodes(g_D,g)
    
    # Do stuff on edges
    g= compute_link_prop(g,g_D)

    g_filt=graph_filter_func(g,g_D)
    #-------------------------------------
    DATA = {}
    DATA['Nber_events'] = sum([g_filt.ep.n_events[v] for v in g_filt.edges()])
    DATA['Nber_links'] = g_filt.num_edges()
    DATA['Nber_nodes'] = g_filt.num_vertices() 
    
    DATA['Proba_rec_event'] = np.mean([g_filt.ep.p_Erec[v] for v in g_filt.edges()])
    DATA['Proba_rec_edge'] = sum([1 for v in g_filt.edges() if g_filt.ep.p_Erec[v]!= 0]) / g_filt.num_edges()
    
    DATA['Burst_nodes'] = np.nanmean([g_filt.vp.burst[v] for v in g_filt.vertices()])
    
    DATA['Burst_edges'] = np.mean([g_filt.ep.burts[e] for e in g_filt.edges()])

#_______________________
    
    results = {}
    results['Method'] = XX
    # PROBA TO HAVE A RECIPROCAL LINK
    results['P(l_rec)'] = DATA['Proba_rec_edge']
    # PROBA TO HAVE A RECIPROCAL EVENT
    results['P(E_rec)'] = DATA['Proba_rec_event']
    # BURSTINESS FROM NODES POINT OF VIEW
    results['B_nodes'] = DATA['Burst_nodes']
    # BURSTINESS FROM EDGES POINT OF VIEW
    results['B_edges'] = DATA['Burst_edges']
        
    
    results_df = pd.DataFrame.from_records([results])
    results_df.set_index("Method", inplace = True)
    
    return results_df

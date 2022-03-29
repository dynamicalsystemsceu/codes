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
from copy import deepcopy
# FUNCTION WHICH CREATES THE AGGREGATED GRAPH: G IS UNDIRECTED AND G_D IS DIRECTED


# def graph_filter_func(g,g_D):
#     # FILTERING
#     #-------------------------------------
    
# #     a. (node filtering) Removing nodes with no reciprocal ecents
# #     g_filt = GraphView(g, vfilt=lambda v: g.vp.n_rec_event[v] > 0.0)   

    
#     # OLDD OKK ??? 
#     #Filtering in and out degree (keep only nodes with degree >=1)
#     g_D_filt = GraphView(g_D, vfilt=lambda v: (v.out_degree()>=1)&(v.in_degree()>=1))
#     g_filt = GraphView(g, vfilt=lambda v: (v in g_D_filt.vertices())==True)

#     # b. (edge filtering) Removing unique edges bw two nodes (ie. if only one event bw two nodes)
#     g_filt = GraphView(g_filt, efilt=lambda e: g_filt.ep.n_events[e] > 4.0)
#     return g_filt
     


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
def compute_edge_n_events(g,g_D):
    properties = list(dict(g_D.edge_properties).keys())
    time_index = properties.index('ts_sec')+2    
    '''
    Note the edge properties will be saved  on g!!! 
        ie. on the aggregate static network
    
    '''
    
    n_events= g.new_edge_property("double") 
    g.ep.n_events= n_events
    
    
    ite= 1
    N=str(g.num_edges())
    
    
    
    for e in g.edges(): # for every edges in the g graph
        
#         print('edge',e)
        
        sys.stdout.write('\r' +'  edges n: '+ str(ite)+'/'+N )

        i = e.source() # node i
        j = e.target() # node j
        
        prop =[g_D.ep[p] for p in dict(g_D.edge_properties).keys()]
        all_edges_i = np.array([e for e in g_D.get_all_edges(i) if e[1] ==j])
        all_edges_j = np.array([e for e in g_D.get_all_edges(j) if e[1] ==i])
        
        ni = len(all_edges_i) 
        nj = len(all_edges_j)

  
        # - Number of events between two nodes
        g.ep.n_events[e] = ni+nj
        ite +=1
        
    return g


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
    g.ep.burst= burts
    
    burst_rec = g.new_edge_property("double") 
    g.ep.burst_rec= burst_rec
    
    burst_no_rec = g.new_edge_property("double") 
    g.ep.burst_no_rec= burst_no_rec
    
    n_events= g.new_edge_property("double") 
    g.ep.n_events= n_events
    
    intertime = g.new_edge_property("vector<double>") 
    g.ep.intertime= intertime
    
    intertime_rec = g.new_edge_property("vector<double>") 
    g.ep.intertime_rec= intertime_rec
    
    intertime_no_rec = g.new_edge_property("vector<double>") 
    g.ep.intertime_no_rec= intertime_no_rec
    
    
    ite= 1
    N=str(g.num_edges())
    
    for e in g.edges(): # for every edges in the g graph
        
#         print('edge',e)
        
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
            
            
#         print('list_events',list_events)
        
        
        # -  Intertimes of the link ij
        intertimes = [list_events[k+1][time_index]-list_events[k][time_index]  for k in range(len(list_events)-1)]
        
        # - Intertimes of the link ij - RECIPROCAL
        intertimes_reciprocal = [list_events[k+1][time_index]-list_events[k][time_index]  for k in range(len(list_events)-1) if list_events[k][0]!=list_events[k+1][0]]
        
#         print('intertimes_reciprocal',intertimes_reciprocal)
        
        # - Intertimes of the link ij - NON RECIPROCAL
        intertimes_noreciprocal = [list_events[k+1][time_index]-list_events[k][time_index]  for k in range(len(list_events)-1) if list_events[k][0]==list_events[k+1][0]]
        
#         print('intertimes_noreciprocal',intertimes_noreciprocal)
        
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
        g.ep.intertime_rec[e] = np.array(intertimes_reciprocal)
        g.ep.intertime_no_rec[e] = np.array(intertimes_noreciprocal)
        
        
        # - Burstiness
        burts = burstiness(intertimes)
        g.ep.burst[e] = burts
        
        # - Burstiness rec
        burts = burstiness(intertimes_reciprocal)
        g.ep.burst_rec[e] = burts
        
#         print('g.ep.burst_rec[e]',g.ep.burst_rec[e])
        
        # - Burstiness no rec
        burts = burstiness(intertimes_noreciprocal)
        g.ep.burst_no_rec[e] = burts  
        

        
        # - Number of events between two nodes
        g.ep.n_events[e] = len(list_events)
        ite +=1
    return g

from statistics import *

from scipy.optimize import curve_fit


from scipy.optimize import curve_fit
def compute_df_ck(df_edges):
    [g,g_D]=to_graph(df_edges,'from','to','t_second','t_minutes','t_hours','t_days')
    prop =[g_D.ep[p] for p in dict(g_D.edge_properties).keys()]
    time_var_index=2    

    N=g_D.num_vertices();
    E=g_D.num_edges();

    g_D_copy=g_D.copy()
    remove_parallel_edges(g_D_copy)
    store_new_yes_no=[];
    store_final_k=[];
    store_current_k=[]


    for i in g_D.vertices():
        current_k=np.zeros(N);

        # take all its event
        all_edges_i = np.array([e for e in g_D.get_all_edges(i,eprops=prop)])
        all_edges_i = sorted(all_edges_i, key = lambda x: x[time_var_index]) # sort according to second, can put time_var_index
        g_ego=Graph(directed=False);
        g_ego.add_vertex(N)
        for e in all_edges_i:
            edge=e[0:2];
            source = edge[0] 
            if source ==i:

                target = edge[1] 
                final_k=g_D_copy.get_out_degrees([source])
                store_final_k.append(final_k[0])

                deg_old=g_ego.get_total_degrees([source])[0]

                store_current_k.append(deg_old)
                g_ego.add_edge(source,target);
                remove_parallel_edges(g_ego)

                deg_new=g_ego.get_total_degrees([source])[0]

                added_=0

                if deg_new>deg_old:
                    added_=1

                store_new_yes_no.append(added_)


    df_ck=pd.DataFrame()
    df_ck['final_k']=store_final_k
    df_ck['current_k']=store_current_k
    df_ck['new_yes_or_no']=store_new_yes_no


    #----------------------------------------------------------------------------
    # Setting up legends and log bins
    #----------------------------------------------------------------------------
    ln_N=int(round(np.log2(N),0))
    bin_s=np.logspace(0,ln_N+1,ln_N+2,base=2)
    leg_=[];xtick_s=[];
    for x in range(len(bin_s)-1):
        leg_.append('$k_{min}$ = '+str(bin_s[x]))
        xtick_s.append(bin_s[x])
    return df_ck,leg_,xtick_s,bin_s

def plot_pk_vs_n_SINGLE(df,bin_s,choice_of_obj):
    store_lines=[]
    #----------------------------------------------------------------------------
    # binning data by degree classes
    #----------------------------------------------------------------------------
    c_k=[]

    store_n=[]
    store_p_k=[]
    
    for i in range(len(bin_s)-1):
    #     new_df=df['new_yes_or_no'][df['new_yes_or_no'].between(bin_s[i], bin_s[i+1], inclusive=False)]
#         df1=df[(bin_s[i] <= df['final_k']) & (df['final_k'] < bin_s[i+1])]
        df1=df[(bin_s[i] <= df['current_k']) & (df['current_k'] < bin_s[i+1])]

        if df1.shape[0]>1:
            store_n.append(bin_s[i])
            store_p_k.append(df1['new_yes_or_no'].mean());


    X=np.array(store_n)
    Y=np.array(store_p_k)
    # FITTING A CURVE
    if len(Y)>3:

        
        x_new=np.linspace(1,1000,1000)
        if choice_of_obj==1:
            popt, _ = curve_fit(objective_1, X,Y)
            a,b=popt
            y_new = objective_1(x_new, a,b);xcol='b'
        else:
            popt, _ = curve_fit(objective_2, X,Y)
            a=popt[0]
            y_new = objective_2(x_new,a);xcol='r'
                

    return popt


def objective_1(x, a,b):
    return a/(x+a)+b
def objective_2(x, a):
    return a/(x+a)

# FUNCTION WHICH CREATES THE AGGREGATED GRAPH: G IS UNDIRECTED AND G_D IS DIRECTED


from copy import deepcopy
def df_filter_func(df_test,n_events_thres): 
    
    df_toreturn = deepcopy(df_test.drop_duplicates())
    
    df_toreturn['couples']=np.where(df_toreturn['from'].astype('int')<df_toreturn['to'].astype('int'),
                                df_toreturn['from'].astype('str')+'_'+df_toreturn['to'].astype('str'),
                                df_toreturn['to'].astype('str')+'_'+df_toreturn['from'].astype('str'))
    
    from_ =np.array(df_toreturn['from'].astype('int'))
    from_unique= np.unique(from_); dict_in_from={key_:'' for key_ in from_unique}
    to_=np.array(df_toreturn['to'].astype('int'))
    to_unique=np.unique(to_);dict_in_to={key_:'' for key_ in to_unique}
    
    df_toreturn['from_has_in'] = np.array([x in dict_in_to.keys() for x in from_]).astype('int')
    df_toreturn['to_has_out'] = np.array([x in dict_in_from.keys()  for x in to_]).astype('int')
    df_toreturn['self_loop']=(from_==to_).astype('int')
    #--------------------------------------------------    
    df_toreturn = df_toreturn.sort_values(by=['couples'])
    couples_ = np.array(df_toreturn['couples'])

    ids= couples_.copy()
    ind=np.where(~(ids[1:ids.shape[0]]==ids[0:ids.shape[0]-1]))
    ind=ind[0]
    career_lens=ind[1:len(ind)]-ind[0:len(ind)-1]
    ind=[-1]+list(ind)
    career_indices=ind + [len(couples_)-1]

    values_ = np.array(career_indices[1:])-np.array(career_indices[0:-1])

    keys_ = np.unique(couples_)

    dict_events = {key_:value_ for key_,value_ in zip(keys_,values_)}

    df_toreturn['n_events'] = df_toreturn['couples'].map(dict_events)

    #---------------------------------------
    #              FILTER step
    #---------------------------------------
    
    df_toreturn=df_toreturn[(df_toreturn['from_has_in']>0)&(df_toreturn['to_has_out']>0)]
    df_toreturn=df_toreturn[df_toreturn['n_events']>=n_events_thres]
    df_toreturn=df_toreturn[df_toreturn['self_loop']==0]
    df_toreturn=df_toreturn[df_test.columns]
    df_toreturn = df_toreturn.sort_values(by=['t_second'])

    return(df_toreturn)

def measures(df_edges,XX,n_events_thres):

    df_edges.drop_duplicates(inplace=True)
    if len(df_edges.columns)==3:
        df_edges.columns=['from', 'to', 't_second']
    df_edges =make_time_col(df_edges,'t_second')

    df_edges= df_filter_func(df_edges,n_events_thres) 
    
    g,g_D = to_graph(df_edges,'from','to','t_second','t_minutes','t_hours','t_days')
    
    
    # Do stuff on nodes
    g = rec_nodes(g_D,g)
    g = burst_rec_nodes(g_D,g)
    g = burst_no_rec_nodes(g_D,g)
    g = burst_nodes(g_D,g)
    
    # Do stuff on edges
    g= compute_link_prop(g,g_D)

    #-------------------------------------
    DATA = table(g)

#_______________________
    
    results = {}
        # PROBA TO HAVE A RECIPROCAL EVENT
    results['P(E_rec)'] = DATA['Proba_rec_event']
    # PROBA TO HAVE A RECIPROCAL LINK
    results['P(l_rec)'] = DATA['Proba_rec_edge']
    # BURSTINESS FROM NODES POINT OF VIEW
    results['B_nodes'] = DATA['Burst_nodes']
    # BURSTINESS FROM EDGES POINT OF VIEW
    results['B_edges'] = DATA['Burst_edges']
    results['data'] = XX
        
    
    results_df = pd.DataFrame.from_records([results])
    results_df.set_index("data", inplace = True)
    return results_df


def table(g_filt):
    
    # TABLE
    #-------------------------------------
    DATA = {}
    DATA['Nber_events'] = sum([g_filt.ep.n_events[v] for v in g_filt.edges()])
    DATA['Nber_links'] = g_filt.num_edges()
    DATA['Nber_nodes'] = g_filt.num_vertices() 
    
    DATA['Proba_rec_event'] = np.nanmean([g_filt.ep.p_Erec[v] for v in g_filt.edges()])
    if g_filt.num_edges()>0:
        DATA['Proba_rec_edge'] = sum([1 for v in g_filt.edges() if g_filt.ep.p_Erec[v]!= 0]) / g_filt.num_edges()
    else:
        DATA['Proba_rec_edge'] = np.nan
    
    DATA['Burst_nodes'] = np.nanmean([g_filt.vp.burst[v] for v in g_filt.vertices()])
    
    DATA['Burst_edges'] = np.nanmean([g_filt.ep.burst[e] for e in g_filt.edges()])

    return(DATA)


def make_time_col(df_edges,t_sec_var):
    df_edges.loc[:,'t_minutes'] = df_edges.loc[:,t_sec_var].apply(lambda x: int(x/60))
    df_edges.loc[:,'t_hours'] = df_edges.loc[:,t_sec_var].apply(lambda x: int(x/3600))
    df_edges.loc[:,'t_days'] = df_edges.loc[:,t_sec_var].apply(lambda x: int(x/86400))
    
    for i_ in [1,2,5,10]:
        for unit_, unit_val_ in zip(['day', 'week', 'month'],[1, 7, 28,]):
            df_edges.loc[:,'t_{}{}'.format(i_,unit_)]= df_edges.loc[:,t_sec_var].apply(
                lambda x: round(x/(86400*i_*unit_val_)))

    return df_edges 

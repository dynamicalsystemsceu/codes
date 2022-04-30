#!/usr/bin/env python
# coding: utf-8

# In[1]:



# df_edges_email = pd.read_table(
#     'email-Eu-core-temporal.txt',header=None,sep=" ")
# df_edges_email.columns = ['from', 'to', 't_second']
# # print(df_edges_email)
# [g,g_D]=to_graph(df_edges_email,'from','to','t_second','t_minutes','t_hours','t_days')
# N=g.num_vertices()
# E=g.num_edges()
# print(N,E)


# In[1]:


# log binned distribution plotting
def plot_loghist(x, bins):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins)
    plt.xscale('log')
    plt.yscale('log')


# In[2]:





# In[46]:


import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from graph_tool.all import *


# power-law random number generator
def rndm(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)




def create_ADAM_power_law_elist(N,E,epsi,g,para1,para2):
    """
    Input: N, number of nodes, integer
           E, number of edges, integer
           epsi, minimum activity, float
           g, exponent of powerlaw (gamma), float
    
    Output: edge list, number of edges
    """
    
    from numpy import random

    #______________________________________________________________________________
    # Setting up node activity 'a' and node attractiviness 'b'
    #______________________________________________________________________________
    a=rndm(epsi, 1, g, size=N) # drawing N values of acitvities "a" from a power law distribution
    # "a" are the node activities which follow the distribution of the out degree
    # assuming a power law out degree distribution with exponent "g-1", 
    # the actitivties also follow a power law distribution with the same exponent


    #PLOTTING activity distribution
        #plot_loghist(a, 50) # visualizing the distribution of node activities 
        #plt.xlabel('activity a');plt.ylabel('N(a) (# nodes with acitvity a)');

    b=a # setting node attractiveness "b" proportional to node activity "a" 
    # following Epidemic spreading on activity-driven networks with attractiveness", PHYSICAL REVIEW E 96, 042310 (2017)

    # Next we set "p", the probability vector of preferential attachment mechanism
    # but before that 
    # we introduce Price's parameter "alpha" which allows us to tune the attachment mechanism from preferential to uniform random 
    alpha = 0
    #for alpha = 0 the pref. attach. mechanism is proportional to the in-degree
    #for aplha >> max(k_in) --> p is a uiniform vector and stubs are attached evenly among nodes

    p = np.array( [(x + alpha)/(sum(b)+ len(b)*alpha ) for x in b])

     # the number of links an active node throws out
    #______________________________________________________________________________
    # GENERATING THE EDGELIST WITH EDGES EQUAL TO THE EMPIRICAL NETWORK 
    #______________________________________________________________________________
    elist = []  # temporal edge list : [ (i,j,t1) , (n,m, t2) ...]
    num_edges=0
    #elist_aggregated =[] # list of all the edges generated during the process, this is not necessary. I used it to verify the goodness of the model
    time = int(1e8)
    # print(active_nodes.shape)

    g = Graph(directed=False)
    g.add_vertex(N)

    for t in range(time): #Loop over time steps, runs as long as needed to generate E edges OR upto 10^8 steps (which is smaller)

        print(num_edges/E,end="\r")
#         print(num_edges,end="\r")
         #print(num_edges)

        remove_parallel_edges(g)
        g_last=g.copy() # graph at last time step
        
        # creating a random array of size N
        y = random.rand(N)
        # Probability of activation proportional to activity of the node
        active_nodes=np.where(y<a)[0] # indices of active nodes

        for i in range(len(active_nodes)): #Loop over active nodes (NOT all nodes)
            source=active_nodes[i]
            degree_node=g_last.get_total_degrees([source])

            if random.rand()<  (para2+(para1/(para1+degree_node))): # with prob=1/(1+k); choose a random neighbour
                target = np.random.choice(range(N),size= 1, p = p ) # select m nodes according to the probability vector p
                target=target[0]
                while target==source:
                    target = np.random.choice(range(N),size= 1, p = p ) 
                    target=target[0]
                elist.append((source, target, t)) 
                g.add_edge(source,target)
                num_edges=num_edges+1

            else:                # else; choose an old neigbour
                neigh=g_last.get_all_neighbors(source)
                if sum(b[neigh]) != 0:
                    p_neigh=np.array( [x/sum(b[neigh]) for x in b[neigh]])
                    if degree_node==1:
                        target=neigh[0]
                        elist.append((source, target, t)) #append the created edges
                        g.add_edge(source,target)                

                    else:
                        degree_node=int(degree_node)

                        for j in list(np.random.choice(range(degree_node),size= 1, p = p_neigh )): # select target from neighbours according to activity
                            target=neigh[j]
                            elist.append((source, target, t)) #append the created edges
                            g.add_edge(source,target)
                    num_edges=num_edges+1

                            
                else:
                    target = np.random.choice(range(N),size= 1, p = p ) # select m nodes according to the probability vector p
                    target=target[0]
                    elist.append((source, target, t)) 
                    g.add_edge(source,target)
                    num_edges=num_edges+1
                    
            

        if num_edges>=E: # stop the Activity driven model if the model has generated as many edges as present in the empirical network 
            break
        

    return elist, num_edges
    #______________________________________________________________________________
    # The algorith runs until it generates edges equal to the number of edges in the empirical network, so big networks may take long time
    #'time' may need to be increased if running for big empirical networks
    #______________________________________________________________________________


    
    
def create_ADAM_power_law_elist_anticorrelated(N,E,epsi,g,para1,para2):
    """
    Input: N, number of nodes, integer
           E, number of edges, integer
           epsi, minimum activity, float
           g, exponent of powerlaw (gamma), float
    
    Output: edge list, number of edges
    """
    
    from numpy import random

    #______________________________________________________________________________
    # Setting up node activity 'a' and node attractiviness 'b'
    #______________________________________________________________________________
    a=rndm(epsi, 1, g, size=N) # drawing N values of acitvities "a" from a power law distribution
    # "a" are the node activities which follow the distribution of the out degree
    # assuming a power law out degree distribution with exponent "g-1", 
    # the actitivties also follow a power law distribution with the same exponent


    #PLOTTING activity distribution
        #plot_loghist(a, 50) # visualizing the distribution of node activities 
        #plt.xlabel('activity a');plt.ylabel('N(a) (# nodes with acitvity a)');

    b=1/a # setting node attractiveness "b" INVERSELY to node activity "a" 
    # following Epidemic spreading on activity-driven networks with attractiveness", PHYSICAL REVIEW E 96, 042310 (2017)

    # Next we set "p", the probability vector of preferential attachment mechanism
    # but before that 
    # we introduce Price's parameter "alpha" which allows us to tune the attachment mechanism from preferential to uniform random 
    alpha = 0
    #for alpha = 0 the pref. attach. mechanism is proportional to the in-degree
    #for aplha >> max(k_in) --> p is a uiniform vector and stubs are attached evenly among nodes

    p = np.array( [(x + alpha)/(sum(b)+ len(b)*alpha ) for x in b])

     # the number of links an active node throws out
    #______________________________________________________________________________
    # GENERATING THE EDGELIST WITH EDGES EQUAL TO THE EMPIRICAL NETWORK 
    #______________________________________________________________________________
    elist = []  # temporal edge list : [ (i,j,t1) , (n,m, t2) ...]
    num_edges=0
    #elist_aggregated =[] # list of all the edges generated during the process, this is not necessary. I used it to verify the goodness of the model
    time = int(1e8)
    # print(active_nodes.shape)

    g = Graph(directed=False)
    g.add_vertex(N)

    for t in range(time): #Loop over time steps, runs as long as needed to generate E edges OR upto 10^8 steps (which is smaller)

        print(num_edges/E,end="\r")
#         print(num_edges,end="\r")
         #print(num_edges)

        remove_parallel_edges(g)
        g_last=g.copy() # graph at last time step
        
        # creating a random array of size N
        y = random.rand(N)
        # Probability of activation proportional to activity of the node
        active_nodes=np.where(y<a)[0] # indices of active nodes

        for i in range(len(active_nodes)): #Loop over active nodes (NOT all nodes)
            source=active_nodes[i]
            degree_node=g_last.get_total_degrees([source])

            if random.rand()<  (para2+(para1/(para1+degree_node))): # with prob=1/(1+k); choose a random neighbour
                target = np.random.choice(range(N),size= 1, p = p ) # select m nodes according to the probability vector p
                target=target[0]
                while target==source:
                    target = np.random.choice(range(N),size= 1, p = p ) 
                    target=target[0]
                elist.append((source, target, t)) 
                g.add_edge(source,target)
                num_edges=num_edges+1

            else:                # else; choose an old neigbour
                neigh=g_last.get_all_neighbors(source)
                if sum(b[neigh]) != 0:
                    p_neigh=np.array( [x/sum(b[neigh]) for x in b[neigh]])
                    if degree_node==1:
                        target=neigh[0]
                        elist.append((source, target, t)) #append the created edges
                        g.add_edge(source,target)                

                    else:
                        degree_node=int(degree_node)

                        for j in list(np.random.choice(range(degree_node),size= 1, p = p_neigh )): # select target from neighbours according to activity
                            target=neigh[j]
                            elist.append((source, target, t)) #append the created edges
                            g.add_edge(source,target)
                    num_edges=num_edges+1

                            
                else:
                    target = np.random.choice(range(N),size= 1, p = p ) # select m nodes according to the probability vector p
                    target=target[0]
                    elist.append((source, target, t)) 
                    g.add_edge(source,target)
                    num_edges=num_edges+1
                    
            

        if num_edges>=E: # stop the Activity driven model if the model has generated as many edges as present in the empirical network 
            break
        

    return elist, num_edges
    #______________________________________________________________________________
    # The algorith runs until it generates edges equal to the number of edges in the empirical network, so big networks may take long time
    #'time' may need to be increased if running for big empirical networks
    #______________________________________________________________________________

    
# In[ ]:





# In[ ]:





# In[55]:


# # # TESTING GROUND; DO NOT DELETE



# # # THESE ARE ARE INPUTS YOU MUST PROVIDE (FROM THE EMPIRICAL NETWORK)
# N=500    # number of nodes in the empirical network 
# E=5000     # number of edges in the empirical network
# epsi=10**-3 # minimum allowed node activity
# g=-1    # g-1 is the exponent of the power law distribution of degrees

# # CALL TO ACVITY DRIVEN MODEL, RETURN TIME ORDERED EDGELIST
# elist,num_edges=create_ADAM_power_law_elist(N,E,epsi,g)
# #print(num_edges)



# In[ ]:





# In[52]:


# from all_functions import *

# Y=measures(elist,'ADA MODEL');


# In[53]:


# print(Y)


# In[129]:


# # 1/(1+degree_node)
# # print(p_neigh)
# # print(source)
# # print(g.get_all_edges([1]))
# # degree_node=g.get_total_degrees([1]);
# # neigh=g.get_all_neighbors(source)
# # print(neigh)
# # print(b[neigh])
# # p_neigh=np.array( [x/sum(b[neigh]) for x in b[neigh]])
# print(p_neigh)

# print(list(p_neigh))

# degree_node=int(degree_node)
# np.random.choice(range(degree_node),size= 1, p = p_neigh )

# # print([x for x in b[neigh]])
# # print(degree_node)
# # print(len(p_neigh))
# # print(p)
# # np.random.choice(range(degree_node),size= 1, p = p_neigh )
# # list(np.random.choice(range(degree_node),size= 1, p = p_neigh ))


# In[128]:





# g = Graph(directed=False)
# g.add_edge(1, 2)
# g.add_edge(2, 3)
# g.add_edge(1, 3)
# g.add_edge(1, 4)
# g.add_edge(5, 1)
# g.add_edge(2, 6)


# print(g.get_total_degrees([1,2,3,4]))
# print(g.get_all_neighbors(1)    )
# neigh=g.get_all_neighbors(2)
# degree_node_=g.get_total_degrees([1000000])
# print(degree_node_[0])
# print(neigh[:])


# In[127]:


# E=5001
# from math import *
# for num_edges in range(E):
#     if ((num_edges*10/E))%1==0:
        
#         #print('finished--',floor(E/(num_edges+1)),' %')
#         print('finished--',(num_edges*100/E),' %')


# In[54]:


#


# In[ ]:





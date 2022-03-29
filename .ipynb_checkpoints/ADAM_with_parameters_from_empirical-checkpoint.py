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


# In[160]:


# log binned distribution plotting
def plot_loghist(x, bins):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins)
    plt.xscale('log')
    plt.yscale('log')


# In[7]:


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


# In[9]:




def create_ADAM_elist(N,E,k_in,k_out,para1,para2):
    from numpy import random

   #______________________________________________________________________________
    # Setting up node activity 'a' and node attractiviness 'b'
    #______________________________________________________________________________
#     a=[];b=[];
    a=np.zeros(N)
    b=np.zeros(N)
    ind=np.random.randint(0, N, size=N)
    
    for i in range(N):
        a[i]=(k_out[ind[i]]/np.sum(k_out));
        b[i]=(k_in[ind[i]]/np.sum(k_in));
    
    
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
        remove_parallel_edges(g)
        g_last=g.copy() # graph at last time step
        
        # creating a random array of size N
        y = random.rand(N)
        # Probability of activation proportional to activity of the node
        active_nodes=np.where(y<a)[0] # indices of active nodes

        for i in range(len(active_nodes)): #Loop over active nodes (NOT all nodes)
            source=active_nodes[i]
            degree_node=g_last.get_total_degrees([source])
            if (para1+degree_node)==0:
                print(para1,para2,degree_node,(para2+(para1/(para1+degree_node))))
            if random.rand()<=(para2+(para1/(para1+degree_node))): # with prob=1/(1+k); choose a random neighbour
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
                        num_edges=num_edges+1

                    else:
                        degree_node=int(degree_node)

                        for j in list(np.random.choice(range(degree_node),size= 1, p = p_neigh )): # select target from neighbours according to activity
                            target=neigh[j]
                            elist.append((source, target, t)) #append the created edges
                            g.add_edge(source,target)                

        #print(num_edges)
        if num_edges>=E: # stop the Activity driven model if the model has generated as many edges as present in the empirical network 
            break


    return elist, num_edges
    #______________________________________________________________________________
    # The algorith runs until it generates edges equal to the number of edges in the empirical network, so big networks may take long time
    #'time' may need to be increased if running for big empirical networks
    #______________________________________________________________________________


# In[ ]:





# In[10]:


# from graph_tool.all import *
# import pickle
# import matplotlib
# import random

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import sys
# import matplotlib.pyplot as plt
# import random
# import math
# import collections
# import csv


# In[11]:



# df_edges_email = pd.read_table(
#     'email-Eu-core-temporal.txt',header=None,sep=" ")
# df_edges_email.columns = ['from', 'to', 't_second']
# # print(df_edges_email)


# In[13]:


# from all_functions import *

# with open('calls_ijt.pickle', 'rb') as handle:
#     calls_data = pickle.load(handle)
# df_edges_calls = pd.DataFrame(calls_data, columns = ['from', 'to', 't_second'])

# # Loading empirical data
# df_edges=df_edges_calls
# [g,g_D]=to_graph(df_edges,'from','to','t_second','t_minutes','t_hours','t_days')
# degree_dist_O = g_D.degree_property_map("out")
# degree_dist_I = g_D.degree_property_map("in")

# k_out=degree_dist_O.a
# k_in=degree_dist_I.a

# # THESE ARE ARE INPUTS WE MUST PROVIDE (FROM THE EMPIRICAL NETWORK)
# N=g.num_vertices()
# E=g.num_edges()

# # CALL TO ACVITY DRIVEN MODEL, RETURN TIME ORDERED EDGELIST
# elist,num_edges=create_ADAM_elist(N,E,k_in,k_out)


# df_edges_ADA_MODEL=pd.DataFrame(list(elist))
# df_edges_ADA_MODEL.columns = ['from', 'to', 't_second']


# In[ ]:





# In[164]:


# # # TESTING GROUND; DO NOT DELETE


# # import matplotlib.pyplot as plt
# # import numpy as np
# # import random
# # import math
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from graph_tool.all import *



# # # THESE ARE ARE INPUTS YOU MUST PROVIDE (FROM THE EMPIRICAL NETWORK)
# N=500      # number of nodes in the empirical network 
# E=5000     # number of edges in the empirical network
# epsi=10**-3 # minimum allowed node activity
# g=-1    # g-1 is the exponent of the power law distribution of degrees

# # # CALL TO ACVITY DRIVEN MODEL, RETURN TIME ORDERED EDGELIST
# elist,num_edges=create_ADAM_power_law_elist(N,E,epsi,g)
# # print(num_edges)



# In[157]:





# In[ ]:





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


# In[ ]:





# In[ ]:





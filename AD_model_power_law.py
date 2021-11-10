#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[1]:


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

# power-law random number generator
def rndm(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)


# In[8]:



def create_AD_power_law_elist(N,E,epsi,g):
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

    m=1 # the number of link an active node throws out
    #______________________________________________________________________________
    # GENERATING THE EDGELIST WITH EDGES EQUAL TO THE EMPIRICAL NETWORK 
    #______________________________________________________________________________
    elist = []  # temporal edge list : [ (i,j,t1) , (n,m, t2) ...]
    num_edges=0
    elist_aggregated = [] # list of all the edges generated during the process, this is not necessary. I used it to verify the goodness of the model
    time = 10**8
    for t in range(time): #Loop over 100 time steps, can be increased if the number of edges in model is less than empirical
        for i in range(N): #Loop over nodes
            if random.random() < a[i]: #Probability of activation proportional to activity of the node
                for _ in list(np.random.choice(range(N),size= m, p = p )): # select m nodes according to the probability vector p
                    elist.append((i, _, t)) #append the created edges
                    elist_aggregated.append((i,_))
                num_edges=num_edges+m;
        if num_edges>=E: # stop the Activity driven model if the model has generated as many edges as present in the empirical network 
            break;

    return elist, num_edges;
    #______________________________________________________________________________
    # The algorith runs until it generates edges equal to the number of edges in the empirical network, so big networks may take long time
    #'time' may need to be increased if running for big empirical networks
    #______________________________________________________________________________


    
    

def create_ADAM_power_law_elist_anticorrelated(N,E,epsi,g):
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

    b=1/a # setting node attractiveness "b" proportional to node activity "a" 
    # following Epidemic spreading on activity-driven networks with attractiveness", PHYSICAL REVIEW E 96, 042310 (2017)

    # Next we set "p", the probability vector of preferential attachment mechanism
    # but before that 
    # we introduce Price's parameter "alpha" which allows us to tune the attachment mechanism from preferential to uniform random 
    alpha = 0
    #for alpha = 0 the pref. attach. mechanism is proportional to the in-degree
    #for aplha >> max(k_in) --> p is a uiniform vector and stubs are attached evenly among nodes

    p = np.array( [(x + alpha)/(sum(b)+ len(b)*alpha ) for x in b])

    m=1 # the number of link an active node throws out
    #______________________________________________________________________________
    # GENERATING THE EDGELIST WITH EDGES EQUAL TO THE EMPIRICAL NETWORK 
    #______________________________________________________________________________
    elist = []  # temporal edge list : [ (i,j,t1) , (n,m, t2) ...]
    num_edges=0
    elist_aggregated = [] # list of all the edges generated during the process, this is not necessary. I used it to verify the goodness of the model
    time = 10**8
    for t in range(time): #Loop over 100 time steps, can be increased if the number of edges in model is less than empirical
        for i in range(N): #Loop over nodes
            if random.random() < a[i]: #Probability of activation proportional to activity of the node
                for _ in list(np.random.choice(range(N),size= m, p = p )): # select m nodes according to the probability vector p
                    elist.append((i, _, t)) #append the created edges
                    elist_aggregated.append((i,_))
                num_edges=num_edges+m;
        if num_edges>=E: # stop the Activity driven model if the model has generated as many edges as present in the empirical network 
            break;

    return elist, num_edges;
    #______________________________________________________________________________
    # The algorith runs until it generates edges equal to the number of edges in the empirical network, so big networks may take long time
    #'time' may need to be increased if running for big empirical networks
    #______________________________________________________________________________


# In[22]:


# # THESE ARE ARE INPUTS YOU MUST PROVIDE (FROM THE EMPIRICAL NETWORK)
# N=5000      # number of nodes in the empirical network 
# E=20     # number of edges in the empirical network
# epsi=10**-3 # minimum allowed node activity
# g=-1    # g-1 is the exponent of the power law distribution of degrees

# # CALL TO ACVITY DRIVEN MODEL, RETURN TIME ORDERED EDGELIST
# elist,num_edges=create_activity_driven_elist(N,epsi,g)
# print(num_edges)


# In[ ]:





# In[ ]:





# In[ ]:





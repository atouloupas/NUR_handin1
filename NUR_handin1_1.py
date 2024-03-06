#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def log_factorial(k):
    """Compute the natural logarithm of a factorial."""
    k_fac = np.linspace(1, k, k)
    
    return np.log(k_fac)


def poisson(lam, k):
    """Compute the poisson distribution through operations in log space."""
    P = k * np.log(lam) - lam - sum(log_factorial(k))
    
    return np.exp(P)


# In[3]:


lam = np.array([1,5,3,2.6,101], dtype=np.float32)
k = np.array([0,10,21,40,200], dtype=np.int32)

with open('NUR_handin1_1.txt', 'a') as f:
    for i in range(len(k)):
        print(f'For lambda = {lam[i]:.1f}, k = {k[i]} --> P = {poisson(lam[i], k[i])}', file=f)




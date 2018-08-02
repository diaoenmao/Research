
# coding: utf-8

# In[2]:

import os
import glob
import csv
import time
import pandas as pd
from   pylab import *
import datetime
import numpy as np
import sys
import math
from   operator import truediv
from   pandas.io.json import json_normalize
import _pickle as cPickle
from   random import *
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import scipy as sp
import pandas as pd
from matplotlib.pyplot import cm
import cyvcf2 as vcf
from cyvcf2 import VCF 
from astropy.table import Table, Column
import h5py
f = h5py.File('dataExportForRelease/wearableDevice/20160503_BIOCHRON_E4.hdf5',  "r")

def read_acc_013(i):
    spe1 = []
    for m in f[f_list[10]][list(f[f_list[10]])[i]][list(f[f_list[10]][list(f[f_list[10]])[i]])[0]]:
        spe1.append(m)
    return spe1
    
#np.concatenate((a,b), axis=0)
def acc_mag(spe):
    acc0 = np.zeros((len(spe),3))
    for i in range(len(spe)):
        acc0[i,0] = spe[i][0]
        acc0[i,1] = spe[i][1]
        acc0[i,2] = spe[i][2]
    acc_mag0 = np.sqrt(acc0[:,0]**2+acc0[:,1]**2+acc0[:,2]**2)
    return acc_mag0

def read_eda_013(i):
    spe1_eda = []
    for m in f[f_list[10]][list(f[f_list[10]])[i]][list(f[f_list[10]][list(f[f_list[10]])[i]])[2]]:
        spe1_eda.append(m)
    return spe1_eda
   

def read_hr_013(i):
    spe1_hr = []
    for m in f[f_list[10]][list(f[f_list[10]])[i]][list(f[f_list[10]][list(f[f_list[10]])[i]])[3]]:
        spe1_hr.append(m)
    return spe1_hr
    
def read_temp_013(i):
    spe1_temp = []
    for m in f[f_list[10]][list(f[f_list[10]])[i]][list(f[f_list[10]][list(f[f_list[10]])[i]])[5]]:
        spe1_temp.append(m)
    return spe1_temp
 
f_list = list(f.keys())


# In[4]:

list(f.values())


# In[5]:

spe013_0 = read_acc_013(0)
acc_mag_013_0 = acc_mag(spe013_0)

spe013_1 = read_acc_013(1)
acc_mag_013_1 = acc_mag(spe013_1)

spe013_2 = read_acc_013(2)
acc_mag_013_2 = acc_mag(spe013_2)

spe013_3 = read_acc_013(3)
acc_mag_013_3 = acc_mag(spe013_3)

spe013_4 = read_acc_013(4)
acc_mag_013_4 = acc_mag(spe013_4)

spe013_5 = read_acc_013(5)
acc_mag_013_5 = acc_mag(spe013_5)

spe013_6 = read_acc_013(6)
acc_mag_013_6 = acc_mag(spe013_6)

spe013_7 = read_acc_013(7)
acc_mag_013_7 = acc_mag(spe013_7)

spe013_8 = read_acc_013(8)
acc_mag_013_8 = acc_mag(spe013_8)

spe013_9 = read_acc_013(9)
acc_mag_013_9 = acc_mag(spe013_9)

spe013_10 = read_acc_013(10)
acc_mag_013_10 = acc_mag(spe013_10)

spe013_11 = read_acc_013(11)
acc_mag_013_11 = acc_mag(spe013_11)

spe013_12 = read_acc_013(12)
acc_mag_013_12 = acc_mag(spe013_12)


spe013_13 = read_acc_013(13)
acc_mag_013_13 = acc_mag(spe013_13)

spe013_14 = read_acc_013(14)
acc_mag_013_14 = acc_mag(spe013_14)

spe013_15 = read_acc_013(15)
acc_mag_013_15 = acc_mag(spe013_15)

spe013_16 = read_acc_013(16)
acc_mag_013_16 = acc_mag(spe013_16)

spe013_17 = read_acc_013(17)
acc_mag_013_17 = acc_mag(spe013_17)

spe013_18 = read_acc_013(18)
acc_mag_013_18 = acc_mag(spe013_18)

spe013_19 = read_acc_013(19)
acc_mag_013_19 = acc_mag(spe013_19)

spe013_20 = read_acc_013(20)
acc_mag_013_20 = acc_mag(spe013_20)

spe013_21 = read_acc_013(21)
acc_mag_013_21 = acc_mag(spe013_21)

spe013_22 = read_acc_013(22)
acc_mag_013_22 = acc_mag(spe013_22)

spe013_23 = read_acc_013(23)
acc_mag_013_23 = acc_mag(spe013_23)


# In[6]:

eda_013_0 = read_eda_013(0)
eda_013_1 = read_eda_013(1)
eda_013_2 = read_eda_013(2)
eda_013_3 = read_eda_013(3)
eda_013_4 = read_eda_013(4)
eda_013_5 = read_eda_013(5)
eda_013_6 = read_eda_013(6)
eda_013_7 = read_eda_013(7)
eda_013_8 = read_eda_013(8)
eda_013_9 = read_eda_013(9)
eda_013_10 = read_eda_013(10)
eda_013_11 = read_eda_013(11)
eda_013_12 = read_eda_013(12)
eda_013_13 = read_eda_013(13)
eda_013_14 = read_eda_013(14)
eda_013_15 = read_eda_013(15)
eda_013_16 = read_eda_013(16)
eda_013_17 = read_eda_013(17)
eda_013_18 = read_eda_013(18)
eda_013_19 = read_eda_013(19)
eda_013_20 = read_eda_013(20)
eda_013_21 = read_eda_013(21)
eda_013_22 = read_eda_013(22)
eda_013_23 = read_eda_013(23)


# In[7]:

hr_013_0 = read_hr_013(0)
hr_013_1 = read_hr_013(1)
hr_013_2 = read_hr_013(2)
hr_013_3 = read_hr_013(3)
hr_013_4 = read_hr_013(4)
hr_013_5 = read_hr_013(5)
hr_013_6 = read_hr_013(6)
hr_013_7 = read_hr_013(7)
hr_013_8 = read_hr_013(8)
hr_013_9 = read_hr_013(9)
hr_013_10 = read_hr_013(10)
hr_013_11 = read_hr_013(11)
hr_013_12 = read_hr_013(12)
hr_013_13 = read_hr_013(13)
hr_013_14 = read_hr_013(14)
hr_013_15 = read_hr_013(15)
hr_013_16 = read_hr_013(16)
hr_013_17 = read_hr_013(17)
hr_013_18 = read_hr_013(18)
hr_013_19 = read_hr_013(19)
hr_013_20 = read_hr_013(20)
hr_013_21 = read_hr_013(21)
hr_013_22 = read_hr_013(22)
hr_013_23 = read_hr_013(23)


# In[8]:

temp_013_0 = read_temp_013(0)
temp_013_1 = read_temp_013(1)
temp_013_2 = read_temp_013(2)
temp_013_3 = read_temp_013(3)
temp_013_4 = read_temp_013(4)
temp_013_5 = read_temp_013(5)
temp_013_6 = read_temp_013(6)
temp_013_7 = read_temp_013(7)
temp_013_8 = read_temp_013(8)
temp_013_9 = read_temp_013(9)
temp_013_10 = read_temp_013(10)
temp_013_11 = read_temp_013(11)
temp_013_12 = read_temp_013(12)
temp_013_13 = read_temp_013(13)
temp_013_14 = read_temp_013(14)
temp_013_15 = read_temp_013(15)
temp_013_16 = read_temp_013(16)
temp_013_17 = read_temp_013(17)
temp_013_18 = read_temp_013(18)
temp_013_19 = read_temp_013(19)
temp_013_20 = read_temp_013(20)
temp_013_21 = read_temp_013(21)
temp_013_22 = read_temp_013(22)
temp_013_23 = read_temp_013(23)


# In[9]:

import pandas as pd
list013 = np.arange(24)
temp_013_list = [temp_013_0,temp_013_1,temp_013_2,temp_013_3,temp_013_4,temp_013_5,temp_013_6,temp_013_7,temp_013_8,temp_013_9,temp_013_10,temp_013_11,temp_013_12,temp_013_13,temp_013_14,temp_013_15,temp_013_16,temp_013_17,temp_013_18,temp_013_19,temp_013_20,temp_013_21,temp_013_22,temp_013_23]     

for i in range(len(temp_013_list)):
    file = pd.DataFrame({'temp_013_%d'%list013[i]:temp_013_list[i]})
    file.to_pickle('temp_013_%d_file.pkl'%list013[i])
    


# In[10]:

import pandas as pd
list013 = np.arange(24)
eda_013_list = [eda_013_0,eda_013_1,eda_013_2,eda_013_3,eda_013_4,eda_013_5,eda_013_6,eda_013_7,eda_013_8,eda_013_9,eda_013_10,eda_013_11,eda_013_12,eda_013_13,eda_013_14,eda_013_15,eda_013_16,eda_013_17,eda_013_18,eda_013_19,eda_013_20,eda_013_21,eda_013_22,eda_013_23]     

for i in range(len(eda_013_list)):
    file = pd.DataFrame({'eda_013_%d'%list013[i]:eda_013_list[i]})
    file.to_pickle('eda_013_%d_file.pkl'%list013[i])
    


# In[11]:

import pandas as pd
list013 = np.arange(24)
hr_013_list = [hr_013_0,hr_013_1,hr_013_2,hr_013_3,hr_013_4,hr_013_5,hr_013_6,hr_013_7,hr_013_8,hr_013_9,hr_013_10,hr_013_11,hr_013_12,hr_013_13,hr_013_14,hr_013_15,hr_013_16,hr_013_17,hr_013_18,hr_013_19,hr_013_20,hr_013_21,hr_013_22,hr_013_23]     

for i in range(len(hr_013_list)):
    file = pd.DataFrame({'hr_013_%d'%list013[i]:hr_013_list[i]})
    file.to_pickle('hr_013_%d_file.pkl'%list013[i])
    


# In[12]:

import pandas as pd
list013 = np.arange(24)
acc_mag_013_list = [acc_mag_013_0,acc_mag_013_1,acc_mag_013_2,acc_mag_013_3,acc_mag_013_4,acc_mag_013_5,acc_mag_013_6,acc_mag_013_7,acc_mag_013_8,acc_mag_013_9,acc_mag_013_10,acc_mag_013_11,acc_mag_013_12,acc_mag_013_13,acc_mag_013_14,acc_mag_013_15,acc_mag_013_16,acc_mag_013_17,acc_mag_013_18,acc_mag_013_19,acc_mag_013_20,acc_mag_013_21,acc_mag_013_22,acc_mag_013_23]     

for i in range(len(acc_mag_013_list)):
    file = pd.DataFrame({'acc_mag_013_%d'%list013[i]:acc_mag_013_list[i]})
    file.to_pickle('acc_mag_013_%d_file.pkl'%list013[i])
    


# In[ ]:




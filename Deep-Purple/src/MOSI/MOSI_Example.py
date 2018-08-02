
# coding: utf-8

# # Example of Obtaining MOSI through CMU-Multimodal Data SDK

# ## Usage
# 
# Please refer to https://github.com/A2Zadeh/CMU-MultimodalDataSDK for more details.

# ### 1 Installation
# In bash:
# 
# git clone git@github.com:A2Zadeh/CMU-MultimodalDataSDK.git
# 
# export PYTHONPATH="./CMU-MultimodalDataSDK:$PYTHONPATH"
# 

# In[1]:


import sys
sys.path.append("./CMU-MultimodalDataSDK/")
from mmdata import Dataloader, Dataset


# ### 2 Merging and Accessing Datasets

# In[2]:


mosi = Dataloader('http://sorena.multicomp.cs.cmu.edu/downloads/MOSI') # feed in the URL for the dataset. 
mosi_visual = mosi.facet()
mosi_text = mosi.embeddings()
mosi_audio = mosi.covarep()
mosi_all = Dataset.merge(mosi_visual, mosi_text)
mosi_all = Dataset.merge(mosi_all, mosi_audio)

print('data ready')
# Let's see what's in the merged dataset

# In[3]:


print(mosi_all.keys())


# ### 3 Loading Train/Validation/Test Splits
# 
# 

# In[4]:


train_ids = mosi.train()
valid_ids = mosi.valid()
test_ids = mosi.test()

print(len(list(train_ids)))
print(len(list(valid_ids)))
print(len(list(test_ids)))

vid = list(train_ids)[0]  
print(vid) # print the first video id in training split


# ### 4 Access Segments and Features

# In[5]:


segment_data = mosi_all['facet'][vid]['3'] # access the facet data in the first video for the 3rd segment
print(len(mosi_all['facet'][vid])) # number of segments

# Check how many features in a segment. Note that number of features may be different from different modalities

# In[6]:


print(len(mosi_all['facet'][vid]['3'])) # number of visual features (30 features per second)
print(len(mosi_all['embeddings'][vid]['3'])) # number of text features (1 feature per word)
print(len(mosi_all['covarep'][vid]['3'])) # number of audio features (100 features per second)


# The format of each feature is "(start_time_1, end_time_1, numpy.array([...]))"

# In[7]:


print(mosi_all['facet'][vid]['3'][0]) # print the first visual feature


# ### 4 Features Alignment between Modalities
# 
# Perform alignment for different modality features. 

# In[8]:


#aligned_text = mosi_all.align('embeddings') # aligning features according to the textual features.
#aligned_audio = mosi_all.align('covarep') # aligning features according to the audio features.
aligned_visual = mosi_all.align('facet') # aligning features according to the visual features.

# assert the features is being aligned!
print(len(aligned_visual['embeddings'][vid]['3']) == len(aligned_visual['facet'][vid]['3']))


# ### 5  Loading Labels

# In[9]:


labels = mosi.sentiments()
print(labels[vid]['3']) # print the labels for the segment


# ### 6 Tutorials
# 
# Install Keras and at least one of the backend (Tensorflow or Theano). Play with `early_fusion_lstm.py` in the `CMU-MultimodalDataSDK/examples`

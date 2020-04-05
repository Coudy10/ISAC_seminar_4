#!/usr/bin/env python
# coding: utf-8

# In[18]:


import nltk
import nltk.corpus
import re
import hashedindex
import sklearn

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np


# In[19]:


fp = open('C:\\Users\\Filip\\Documents\\School\\ING\\ISAC\\cvika\\abstracts.txt') 
text = fp.read()

documents = text.splitlines()

search = 'machine learning text'

vocabulary = word_tokenize(search)

vocabulary


# In[22]:


pipe = Pipeline([('count', CountVectorizer(vocabulary = vocabulary)),
                 ('tfid', TfidfTransformer())]).fit(documents)
pipe['count'].transform(documents).toarray()


# In[23]:


pipe['tfid'].idf_


# In[24]:


pipe.transform(documents).shape


# In[ ]:





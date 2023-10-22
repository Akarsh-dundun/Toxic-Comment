#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np 
import pandas as pd
import seaborn as sns


# In[2]:


#pip install --upgrade spacy


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import spacy
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# In[4]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#print(train)
train=train.drop('id', axis=1)
print(train)
test=test.drop('id', axis=1)



# In[5]:


total_null = train.isnull().sum()
print(total_null)


# In[6]:


total_null = test.isnull().sum().sort_values(ascending = False)
print(total_null)


# In[7]:


cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']
print(train[cols_target].sum())


# In[51]:


x=train.iloc[:,1:]
#print(x)
y=train.iloc[:,1:].sum()
#print(y)
rowsums=train.iloc[:,2:].sum(axis=1)
#print(rowsums)
train['clean']=(rowsums==0)
train['clean'].sum()
print("Total comments = ",len(train))
print("Total clean comments = ",train['clean'].sum())
print("Total tags =",x.sum())




# In[9]:



categories = ['Obscene', 'Insult', 'Toxic', 'Severe_toxic', 'Identity_hate', 'Threat', 'Total Clean Comments']
counts = [8449, 7877, 15294, 1595, 1405, 478, 143346]

# Create the bar plot

#xvals=range(len(counts))
#plt.bar(xvals, counts)
#plt.tick_params(bottom=False)
#plt.xticks(xvals,['Obscene', 'Insult', 'Toxic', 'Severe_toxic', 'Identity_hate', 'Threat', 'Total Clean Comments'], rotation=30
  #,horizontalalignment='right')

#plt.gca().spines['top'].set_visible(False)
#plt.gca().spines['right'].set_visible(False)
# Create the bar plot
plt.bar(categories, counts)

# Labeling the axes and giving a title
plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Counts of Different Categories')

# Show the plot
plt.show()


#plt.bar(categories, counts)


# In[10]:


# Let's look at the character length for the rows in the training data and record these
train['char_length'] = train['comment_text'].apply(lambda x: len(str(x)))
# look at the histogram plot for text length
sns.set()
train['char_length'].hist()
plt.show()


# In[11]:


data = train[cols_target]
colormap = plt.cm.plasma
plt.figure(figsize=(7,7))
plt.title('Correlation of features & targets',y=1.05,size=14)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,
           linecolor='white',annot=True)


# In[12]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


# In[52]:


train['comment_text'] = train['comment_text'].map(lambda com : clean_text(com))
test['comment_text'] = test['comment_text'].map(lambda com : clean_text(com))


# In[53]:


train


# In[54]:


train = train.drop('clean',axis=1) #we used char_length for plotting distribution now lets remove it


# In[58]:


#X = train.comment_text
#test_X = test.comment_text
x = train.iloc[:, :1].values
y = train.iloc[:, 1:].values



# In[65]:


categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train, y_test = train_test_split(train, random_state=42, test_size=0.2, shuffle=True)
x_train = y_train.comment_text
x_test = y_test.comment_text


# In[46]:


train.head()


# In[72]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(max_features=7500,stop_words='english')
vect


# In[73]:


# learn the vocabulary in the training data, then use it to create a document-term matrix
X_dtm = vect.fit_transform(x_train)
# examine the document-term matrix created from X_train
pd.DataFrame(X_dtm.todense())


# In[74]:


# transform the test data using the earlier fitted vocabulary, into a document-term matrix
test_X_dtm = vect.transform(x_test)
# examine the document-term matrix from X_test
pd.DataFrame(test_X_dtm.todense())


# In[ ]:


from sklearn.naive_bayes import 


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# # Importing DataSet

# In[2]:


df=pd.read_csv(r"C:\Users\HP\Downloads\titanic\train.csv")


# In[ ]:


# EDA


# In[3]:


df.head(5)


# In[5]:


df.shape


# In[6]:


df.info()


# In[8]:


df.isna().sum()


# In[ ]:





# # Deleting the unknown column

# In[9]:


df = df.drop("Unnamed: 12", axis=1)


# In[10]:


df = df.drop("Unnamed: 13", axis=1)


# In[11]:


df = df.drop("Unnamed: 14", axis=1)


# In[13]:


pd.DataFrame(df)


# In[41]:


df.head(10)


# In[14]:


df.duplicated().sum()


# # visualization

# In[16]:


sns.countplot(data=df,x="Sex")


# In[21]:


df['Pclass'].value_counts().plot(kind='pie', autopct='%.2f%%', explode=[0,0,0.05])
plt.title('pclass counts')
plt.xlabel('pclass', weight='bold')
plt.ylabel('counts', weight='bold')
plt.show()


# In[22]:


sns.countplot(data=df,x='SibSp')


# In[24]:


plt.pie(df['Survived'].value_counts(), autopct='%.2f%%', labels=[0, 1])
plt.title('survived ratio')
plt.xlabel('survived')
plt.ylabel('count')
plt.show()


# In[26]:


sns.histplot(data=df,x="Age")


# In[ ]:


# Visualization Based on Gender ,whether Individuals Survived or not


# In[28]:


gender=sns.FacetGrid(df,col="Survived")

gender.map(plt.hist,"Sex")


# In[ ]:





# # Data Processing

# In[30]:


df.isna().sum()


# 'cabin' column have a large number of dataset having a miising value so we can drop that

# In[33]:


df.Sex.replace({"male":1,"female":0},inplace=True)


# In[34]:


df.head()


# In[35]:


df.Embarked.replace({'S':1,'C':2,'Q':3},inplace=True)


# In[36]:


df.head(5)


# In[37]:


df.corr()


# In[40]:


sns.heatmap(df.corr(),annot=True,cmap="winter")


# In[ ]:





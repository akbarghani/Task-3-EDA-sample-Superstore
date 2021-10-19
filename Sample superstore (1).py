#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


df=pd.read_csv("C:/Users/Faisal Khan/Desktop/Spark Intern/SampleSuperstoreCSV.csv" )


# In[3]:


df


# In[4]:


df=df.drop(labels='13',axis=1)


# In[5]:


df


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.info()


# In[9]:


df.isnull()


# In[10]:


df.isnull().sum()


# In[11]:


df.sample(9)


# In[12]:


print(df.info())   


# # SUMMARY OF THE DATA

# In[13]:


df.describe()


# #SHAPE OF DATA

# In[14]:


df.shape


# #Data type of columns

# In[15]:


df.dtypes


# In[16]:


# Columns of DataSet


# In[17]:


df.columns


# In[18]:


#now tCheck the dataset for duplicate and dropping element¶


# In[19]:


df.duplicated().sum()


# In[20]:


df.drop_duplicates()


# #the number of unique values in each column
# 

# In[21]:


df.nunique()


# Now the correlationbetween the integer type variables 

# In[22]:


df.corr()


# In[23]:


plt.figure(figsize=(16,8))
plt.bar(df['Sub-Category'],df['Category'], data=df)
plt.ylabel("Category")
plt.xlabel("SubCategory")
plt.show()


# In[24]:


print(df['State'].value_counts())
plt.figure(figsize=(15,8))
sns.countplot(x=df['State'])
plt.xticks(rotation=90)
plt.show()


# In[25]:


print(df['Sub-Category'].value_counts())
plt.figure(figsize=(12,6))
sns.countplot(x=df['Sub-Category'])
plt.xticks(rotation=90)
plt.show()


# In[26]:


fig,axes = plt.subplots(1,1,figsize=(9,6))
sns.heatmap(df.corr(), annot= True)
plt.show()


# In[27]:


sns.countplot(x=df['Segment'])


# In[28]:


sns.countplot(x=df['Category'])


# In[29]:


sns.countplot(x=df['Country'])


# In[30]:


sns.countplot(x=df['Region'])


# In[31]:


plt.figure(figsize = (10,4))
sns.lineplot('Discount', 'Profit', data = df, color = 'b', label= 'Discount')
plt.legend()


# In[32]:


df.hist(bins=50 ,figsize=(20,15))
plt.show()


# In[33]:


figsize=(15,10)
sns.pairplot(df,hue='Sub-Category')


# In[34]:


print(df['Sales'].describe())
plt.figure(figsize = (9 , 8))
sns.distplot(df['Sales'], color = 'b', bins = 100, hist_kws = {'alpha': 0.4});


# # Scatterplot

# In[35]:


fig, ax = plt.subplots(figsize = (10 , 6))
ax.scatter(df["Sales"] , df["Profit"])
ax.set_xlabel('Sales')
ax.set_ylabel('Profit')
plt.show()


# In[39]:


Q1 = df.quantile(q = 0.25, axis = 0, numeric_only = True, interpolation = 'linear')

Q3 = df.quantile(q = 0.75, axis = 0, numeric_only = True, interpolation = 'linear')
IQR = Q3 - Q1

print( "The Inter Quartile Range of RESPECTIVE VARIABLE IS GIVEN AS",IQR)


# In[40]:


fig, axes = plt.subplots(figsize = (10 , 10))

sns.boxplot(df['Profit'])


# In[41]:


fig, axes = plt.subplots(figsize = (10 , 10))

sns.boxplot(df['Discount'])


# In[42]:


x = df.iloc[:, [9, 10, 11, 12]].values
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0).fit(x)
    wcss.append(kmeans.inertia_)

sns.set_style("whitegrid") 
sns.FacetGrid(df, hue ="Sub-Category",height = 6).map(plt.scatter,'Sales','Quantity')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:





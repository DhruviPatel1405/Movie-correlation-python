#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[2]:


import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)


# In[4]:


df = pd.read_csv('movies.csv')


# In[5]:


df.info


# In[6]:


df.head()


# In[8]:


print(df.describe())


# In[9]:


print(df.isnull().sum())


# In[10]:


df.shape


# In[15]:


df.dropna(inplace=True)


# In[16]:


df.shape


# In[17]:


print(df.isnull().sum())


# In[19]:


df.dtypes


# In[25]:


df['votes']= df['votes'].astype('int64')


# In[27]:


df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')


# In[28]:


df.dtypes


# In[29]:


pd.isnull(df).sum()


# In[30]:


df.columns


# In[31]:


df.describe(include='object')


# In[32]:


df.head()


# In[78]:


df = df.sort_values(by=['gross'], inplace=False,ascending=False)


# In[79]:


pd.set_option('display.max_row',None)


# In[53]:


#drop duplicates 
df.drop_duplicates()


# In[52]:


print(df.duplicated().sum())


# In[54]:


#budgest high correlation
#company high coreelation


# In[61]:


#scatter plot with budge vs gross
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earings')
plt.ylabel('Buget for Film')
plt.show()


# In[59]:


df.head()


# In[63]:


#plot buggest vs gross using seaborn 

sns.regplot(x='budget',y='gross',data=df,scatter_kws={"color": "red"},line_kws={"color": "blue"})


# In[65]:


df.corr(method='pearson') # pearson , kendall , spearman


# In[69]:


correlation_matrix = df.corr(method='pearson')

sns.heatmap(correlation_matrix,annot=True)
plt.title('Correlation Matric For Numeric Features')
plt.xlabel('Movie features')
plt.ylabel('Movie features')
plt.show()


# In[70]:


df.head()


# In[76]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

        
df_numerized    


# In[81]:


df 


# In[82]:


correlation_matrix = df_numerized.corr(method='pearson')

sns.heatmap(correlation_matrix,annot=True)
plt.title('Correlation Matric For Numeric Features')
plt.xlabel('Movie features')
plt.ylabel('Movie features')
plt.show()


# In[83]:


df_numerized.corr()


# In[85]:


correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()


# In[86]:


corr_pairs


# In[89]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs 


# In[90]:


high_corr = sorted_pairs[(sorted_pairs)>0.5]
high_corr


# In[ ]:





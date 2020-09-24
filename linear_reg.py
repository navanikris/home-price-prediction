#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("C:/Users/user/Desktop/data.csv")
df2 = pd.read_csv("C:/Users/user/Desktop/Canada.csv")


# In[3]:


print(df.head())
print(df2.head())


# In[4]:


plt.xlabel("area")
plt.ylabel("price")
plt.scatter(df.area,df.price,color='red',marker='*')


# In[5]:


plt.xlabel("Year")
plt.ylabel("Income")
plt.scatter(df2.year,df2.Income,color='red',marker='+')


# In[6]:


reg = linear_model.LinearRegression()
reg.fit(df2[['year']],df2.Income)


# In[7]:


reg.predict([[2020]])


# In[8]:


reg1 = linear_model.LinearRegression()
reg1.fit(df[['area']],df.price)


# In[9]:


reg1.predict([[2020]])


# In[10]:


reg1.predict([[3300]])


# In[11]:


#Linear regression with multiple variables also known as multivariate regression


# In[12]:


df1 = pd.read_csv("C:/Users/user/Desktop/data2.csv")


# In[13]:


df1.head()


# In[14]:


#given these home prices find out price of a home that has,
#3000 sqrft area,3 bedrooms,40 years old
#2500 sqrft area,4 bedrooms,5 years old


# In[15]:


#calculate median
import math
median = math.floor(df1.bedroom.median())
median


# In[16]:


df1.bedroom = df1.bedroom.fillna(median)
df1


# In[17]:


reg2 = linear_model.LinearRegression()
reg2.fit(df1[['area','bedroom','age']],df1.price)


# In[18]:


reg2.predict([[3000,3,40]])


# In[19]:


reg2.predict([[2500,4,5]])


# In[20]:


reg2.coef_


# In[21]:


reg2.intercept_


# In[22]:


#y = mx+c
#y = m1*x1 + m2*x2 + m3*x3 +c
'''
m - co.efficent
x - independent variable
c - intercept
'''


# In[23]:


df3 = pd.read_csv("C:/Users/user/Desktop/emp.csv")


# In[24]:


df3.head()


# In[25]:


df3['test_score(out of 10)']


# In[26]:


med= math.floor(df3['test_score(out of 10)'].median())
df3['test_score(out of 10)']  = df3['test_score(out of 10)'].fillna(med) 


# In[27]:


df3


# In[28]:


df3.experience = df3.experience.fillna('zero')


# In[29]:


df3


# In[1]:


pip install word2number  --user


# In[30]:


from word2number import w2n


# In[32]:


df3.experience =  df3.experience.apply(w2n.word_to_num)


# In[33]:


df3.head()


# In[34]:


reg3 = linear_model.LinearRegression()
reg3.fit(df3[['experience','test_score(out of 10)','interview_score(out of 10)']],df3['salary($)'])


# In[35]:


reg3.predict([[2,9,6]])


# In[36]:


reg3.predict([[12,10,10]])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





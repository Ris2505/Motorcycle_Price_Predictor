#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd;


# In[6]:


bike=pd.read_csv('Used_Bikes.csv')


# In[7]:


bike.head()


# In[8]:


bike.shape


# In[9]:


bike.info()


# In[10]:


bike['age'].unique()


# In[11]:


bike['price'].unique()


# In[12]:


bike['kms_driven'].unique()


# In[13]:


bike['bike_name']


# In[16]:


bike['bike_name']=bike['bike_name'].str.split(' ').str.slice(0,3).str.join(' ')


# In[17]:


bike['bike_name']


# In[18]:


bike


# In[19]:


bike.describe()


# In[21]:


bike[bike['price']>1e6]


# In[70]:


x=bike.drop(columns='price')


# In[71]:


y=bike['price']


# In[72]:


x=x.drop(columns='owner')
x=x.drop(columns='power')


# In[73]:


x


# In[74]:


y


# In[101]:


from sklearn.model_selection import train_test_split
x_train , x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[102]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# In[103]:


x_test


# In[104]:


x_train


# In[105]:


ob=OneHotEncoder()
ob.fit(x[['bike_name','city','kms_driven','age','brand']])


# In[106]:


ob.categories_


# In[119]:


column_trans=make_column_transformer((OneHotEncoder(categories=ob.categories_),['bike_name','city','kms_driven','age','brand']),remainder='passthrough')


# In[120]:


linearregression=LinearRegression()


# In[121]:


pipe=make_pipeline(column_trans,linearregression)
pipe.fit(x,y)


# In[122]:


y_pred=pipe.predict(x_test)


# In[126]:


r2_score(y_test,y_pred)


# In[124]:


y_test


# In[127]:


import pickle


# In[129]:


pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))


# In[ ]:





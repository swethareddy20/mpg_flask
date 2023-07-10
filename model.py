#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import pickle


# In[27]:



# In[28]:


mpg_df=pd.read_csv("Auto MPG Reg.csv.csv")


# In[29]:


mpg_df.horsepower=mpg_df.horsepower.fillna(mpg_df.horsepower.median())


# In[30]:


y=mpg_df.mpg
X=mpg_df.drop('mpg',axis=1)


# In[31]:


from sklearn.linear_model import LinearRegression


# In[41]:


reg=LinearRegression()


# In[42]:


reg.fit(X,y)


# In[43]:


reg.score(X,y)


# In[44]:


regpredict=reg.predict(X)


# In[45]:


from sklearn.metrics import mean_squared_error


# In[46]:


np.sqrt(mean_squared_error(y,regpredict))


pickle.dump(reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))





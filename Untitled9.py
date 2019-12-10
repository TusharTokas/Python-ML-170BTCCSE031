#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets , linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data=pd.read_csv(r"C:\Users\tussh\Downloads\amazon.csv")
#data
'''x=data.iloc[:10,1]
#x
#(['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
#dib=datasets.load_diabetes()
#print(dib.keys())
m_cur = b_cur = 0
iterations = 1000
n = len(x)
learning_rate = 0.001
for i in range(iterations):
 y_pd = m_cur * x + b_cur
 #print(y_pd)
 cost = (1/n) * np.sum((y-y_pd)**2)
 #print(cost)
 md = -(2/n)*sum(x*(y-y_pd)) #md here is slope
 bd = -(2/n)*sum(y-y_pd)     #bd here is intercept i.e.c
 m_cur = m_cur - learning_rate * md
 b_cur = b_cur - learning_rate * bd

print("y predicted")
print("y_pd")
print(" value of m  ", m_cur)
print("value of c is" ,b_cur)
plt.scatter(x,y,color='blue')
plt.plot(x,y_pd,color='green')
plt.title('state vs numer')
plt.xlabel('state')
plt.ylabel('number')
'''


# In[55]:


x=data[['state']]
y=data[['number']]
#y.head(),x.head()
#x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.30, random_state=0)
#x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[56]:


#regressor=LinearRegression()
#regressor.fit(x_train, y_train)
#y_pred=regressor.predict(x_test)


# In[39]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[48]:


from sklearn.linear_model import LogisticRegression
log_model =LogisticRegression()
#log_model.fit(x_train,y_train)


# In[ ]:





# In[ ]:





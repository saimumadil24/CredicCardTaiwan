#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('UCI_Credit_Card.csv')
data


# In[3]:


data.isnull().sum()


# In[4]:


data.describe()


# In[5]:


data['SEX'].value_counts()


# In[6]:


data['EDUCATION'].value_counts()


# In[7]:


data['EDUCATION']=data['EDUCATION'].replace([5,6,0],4)


# In[8]:


data['EDUCATION'].value_counts()


# In[9]:


data['MARRIAGE'].value_counts()


# In[10]:


data['MARRIAGE']=data['MARRIAGE'].replace(0,3)


# In[11]:


data['MARRIAGE'].value_counts()


# In[12]:


plt.boxplot(data['LIMIT_BAL'])
plt.show()


# In[13]:


data['LIMIT_BAL'].nlargest(5)


# In[14]:


data=data.rename(columns={'default.payment.next.month':'IsDefaulter'})
data


# In[15]:


data=data.drop('ID',axis=1)
data


# In[16]:


x=data.drop('IsDefaulter',axis=1)
y=data['IsDefaulter']


# In[17]:


from sklearn.svm import SVC


# In[52]:


sv=SVC(gamma='auto')


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)


# In[20]:


#sv.fit(x_train,y_train)


# In[21]:


#sv.predict(x_test)


# In[22]:


#sv.score(x_test,y_test)


# In[23]:


from sklearn.ensemble import RandomForestClassifier


# In[24]:


rfc=RandomForestClassifier()


# In[25]:


rfc.fit(x_train,y_train)


# In[26]:


rfc.score(x_test,y_test)


# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[29]:


lr.score(x_test,y_test)


# In[30]:


data['SEX']=data['SEX'].replace({1:'Male',2:'Female'})


# In[31]:


data['EDUCATION']=data['EDUCATION'].replace({1:'graduate school',2:'university',3:'school',4:'others'})


# In[32]:


data['MARRIAGE']=data['MARRIAGE'].replace({1:'married',2:'single',3:'others'})


# In[33]:


data['AGE'].max()


# In[34]:


data['AGE'].min()


# In[35]:


data['AgeGroup']=pd.cut(data['AGE'],bins=[20,35,60,80],labels=['PreMature','Mature','Eldest'],right=False)


# In[36]:


data


# In[37]:


data=pd.get_dummies(data,columns=['SEX','EDUCATION','MARRIAGE','AgeGroup'],drop_first=True)


# In[38]:


data


# In[46]:


X=data.drop('IsDefaulter',axis=1)


# In[47]:


Y=data['IsDefaulter']


# In[48]:


data['IsDefaulter'].isnull().sum()


# In[49]:


X.isnull().sum()


# In[50]:


X


# In[51]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1)


# In[53]:


sv.fit(X_train,Y_train)


# In[54]:


sv.predict(X_test)


# In[58]:


sv.score(X_test,Y_test)


# In[59]:


rfc.fit(X_train,Y_train)


# In[60]:


rfc.score(X_test,Y_test)


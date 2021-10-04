#!/usr/bin/env python
# coding: utf-8

# # Gradient Descent

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[46]:


z = np.linspace(-10,10,100)
def f(x):
    return 0.5*x**4 - 3*x**3
plt.ylim(-100,100)
plt.xlim(-5,7)
plt.plot(z,f(z))


# In[87]:


def gd(gamma,x,b): # B 는 추진력 
    dx = 0.001
    df = (f(x+dx)-f(x))/dx # 기울기
    return x-gamma*df+b


# In[88]:


x0=-2
x1 = [x0]
y1 = [f(x0)]
gam=0.01

for i in range(1000):
    x0=gd(gam,x0,0.1)
    x1=np.append(x1,x0)
    y1=np.append(y1,f(x0))
    
plt.ylim(-100,100)
plt.xlim(-5,7)
plt.plot(z,f(z))
plt.plot(x1,y1,'o')


# # Diabetes (regression)¶

# In[1]:


from sklearn import datasets


# In[2]:


diabetes=datasets.load_diabetes()
diabetes.keys()


# In[3]:


print(diabetes.data) # 데이터 항목에서 제대로된 정보를 얻을수 없음


# In[4]:


print(diabetes.feature_names)


# In[ ]:





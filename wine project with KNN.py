#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns 
import matplotlib.pyplot as plt


# In[2]:


data = load_wine()


# In[3]:


df = pd.DataFrame(data['data'],columns = data['feature_names'])

df['target'] = data['target']
df


# In[5]:


x_train, x_test, y_train, y_test = train_test_split(data['data'],data['target'],random_state=42,test_size=0.2)

model = load_wine()
model.fit(x_train,y_train)


# In[6]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns 
import matplotlib.pyplot as plt


# In[7]:


data = load_wine()


# In[8]:


df = pd.DataFrame(data['data'],columns = data['feature_names'])

df['target'] = data['target']
df


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(data['data'],data['target'],random_state=42,test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)


# In[10]:


model.fit(x_train,y_train)


# In[11]:


y_pred = model.predict(x_test)


# In[12]:


model.score(x_test,y_test)


# In[13]:


print(classification_report(y_test,y_pred))


# In[15]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[16]:


sns.heatmap(cm,annot=True,fmt= '.2g')
plt.show()


# In[17]:


import time

max_score = 0
all_score = []
all_k = []
for i in range(1,16):
    x_train, x_test, y_train, y_test = train_test_split(data['data'],data['target'],random_state=32,test_size=0.2)
    
    model = KNeighborsClassifier(n_neighbors=i)
    
    model.fit(x_train,y_train)
    score = model.score(x_test,y_test)
    
    if score>max_score:
        max_score = score
        
        
        print(f'Value of K is: {i} and score is {score}')
        display(clear=True)
    all_score.append(score)
    all_k.append(i)
    time.sleep(1)


# In[18]:


import numpy as np


# In[19]:


plt.plot(range(1,16),all_score)

k_index = np.argmax(all_score)

plt.annotate(text = f'max score: {round(max(all_score),2)}, k is {k_index}',
            xy = (k_index,max(all_score)))
plt.show()


# In[ ]:





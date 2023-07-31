#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from IPython.core.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
'exec(%matplotlib inline)'

data = pd.read_csv('winequality-red.csv')


# In[16]:


data.describe()


# In[17]:


data.head()


# In[18]:


data.isnull().any()
values = {'fixed acidity': 0}
data = data.fillna(value=values)

columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
               'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
X = data[columns].values
y = data['quality'].values


# In[19]:


plt.figure(figsize=(10, 10))
plt.tight_layout()
seabornInstance.distplot(data['fixed acidity'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

X = data.values
coeff_df = pd.DataFrame(regressor.coef_, columns, columns=['Coefficient'])
display(coeff_df)


# In[20]:


y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dataDisplay = df.head(20)
display(dataDisplay)


# In[21]:


dataDisplay.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.4', color='pink')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='blue')
plt.show()


# In[ ]:





# In[ ]:





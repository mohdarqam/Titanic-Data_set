#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
data = pd.read_csv(r'C:\Users\mohda\Downloads\titanic_data\Cleaning-Titanic-Data-master\titanic_original.csv')


# In[2]:


data.head()


# # Data cleaning:
# ### for Nan values we use visualisation technique to see that.
# ### heatmap can show the all the nan values.
# 

# In[4]:


sns.heatmap(data.isnull(),yticklabels = False,cbar = False,cmap= 'viridis')


# ### Columns which has too many Nan value we will drop it because we dont have enough information.
# 

# In[5]:


data.drop(['cabin'],axis = 1, inplace = True)


# In[6]:


data.drop(['home.dest'],axis =1, inplace=True)


# In[7]:


data.drop(['boat'],axis =1, inplace=True)


# In[8]:


data.drop(['body'],axis =1, inplace=True)


# In[9]:


#yellow part is Nan values
sns.heatmap(data.isnull(),yticklabels = False,cbar = False,cmap= 'viridis')


# In[10]:


sns.countplot('survived',data=data)


# ## Count plot to see how many survived.
# ### we can see that from the graph survial rate is less.

# In[11]:


sns.set_style('whitegrid')
sns.countplot(x='survived',hue='sex',data=data,palette='viridis')


# ## Count plot of survived with respect to sex.
# ### we number of spouse and sibling in the ship can see that female survived more than the male

# In[12]:


sns.countplot(x='survived',hue='pclass',data=data,palette='rainbow')


# ## Countplot of survived with respect to pclass(passenger class).
# ### from the graph we can see that passenger from fisrt class survived more than the other two class.
# ### least survival count is from the passenger is from the third class

# In[13]:


sns.distplot(data['age'].dropna(),kde=False,color='darkred',bins=40)


# ## finding age of passenger in the ship we use the histogram.
# ### from the graph we can see that age group from 18 to 33 approx are maximum in the graph.
# ### young people ratio is more than that other.

# In[14]:


sns.countplot('sibsp',data=data,palette='rainbow')


# ### Number of spouse and sibling in the ship.

# In[15]:


data['fare'].hist(bins= 50)


# ### Maximum number of people have third class of ticket fare which is less tha 50 pound.

# # Above we did not clean data of Age.
# ## now we replace the Nan value from mean value of class wise avegrage age

# In[16]:


plt.figure(figsize=(12,7))
sns.boxplot(x='pclass',y='age',data=data)


# ### Above box graph we can see that :
# #### first class average age is 37
# #### second class average age is 29
# #### third class avegrage age is 26

# In[17]:


def impute_value(cols):
    pclass = cols[0]
    age = cols[1]
    
    if(pd.isnull(age)):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 26
    else:
        return age
    
        
    


# In[18]:


data['age'] = data[['age','pclass']].apply(impute_value,axis=1)


# In[19]:



sns.heatmap(data.isnull(),cbar=False,yticklabels = False,cmap = 'rainbow')


# ## Now our data is almost clean.

# In[20]:


data.head()


# ## Now for the categorical data we will use the feature engineering.
# ### Also we dont want the 'name'.
# 

# In[21]:


sex = pd.get_dummies(data['sex'],drop_first=True)


# In[22]:


embarked=pd.get_dummies(data['embarked'],drop_first=True)


# In[23]:


embarked.drop('Q',axis=1,inplace=True)


# In[24]:


data.drop(['name','sex','embarked'],axis=1,inplace=True)


# In[25]:


data.drop('ticket',axis=1,inplace=True)


# In[31]:


data.dropna(inplace=True)
data.isnull().sum()


# In[27]:


data = pd.concat([data,sex,embarked],axis=1)
data.head()


# ### From above we can see that our data is ready for the training and testing.

# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train,X_test,y_train,y_test = train_test_split(data.drop('survived',axis=1),
                                                 data['survived'],test_size=1/3,
                                                 random_state=0)


# In[34]:


from sklearn.linear_model import LogisticRegression
p = LogisticRegression()
p.fit(X_train,y_train)
pred = p.predict(X_test)


# In[35]:


p.score(X_train,y_train)


# ## Model has 79.2% accuracy.

# In[36]:


from sklearn.metrics import mean_squared_error, r2_score


# In[37]:


rmse = np.sqrt(mean_squared_error(y_test,pred))


# In[38]:


rmse


# ## Model has 0.46 maximum error.

# In[39]:


new = pd.DataFrame({'Survived':y_test,'Pre_Survived':pred})


# In[40]:


new.head()


# In[ ]:


-


#!/usr/bin/env python
# coding: utf-8

# # project
# 
# Use the "Run" button to execute the code.

# ## Install the Libraries

# In[5]:


get_ipython().system('pip install pandas-profiling numpy matplotlib seaborn --quiet')


# In[1]:


get_ipython().system('pip install opendatasets scikit-learn --quiet --upgrade')


# # Downloading the Dataset and Extracting

# In[7]:


import opendatasets as od
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import os


# In[8]:


od.download('https://drive.google.com/file/d/1Ljyhi1NcdRvxTizCJ3G8facwKnpmruDO/view?usp=sharing')


# In[9]:


data_df=pd.read_csv('Weather_data.csv')


# ## Dataset

# In[10]:


data_df


# ### Finding the missing values in given dataset

# In[11]:


data_df.info()


# ### Finding the input columns and Target column for Prediction

# In[12]:


target_cols=' _tempm'


# In[13]:


input_cols1=list(data_df.columns)[0:11]
input_cols2=list(data_df.columns)[12:20]
input_cols=input_cols1 + input_cols2
input_cols


# ### Input Data Frame

# In[14]:


inputs_df = data_df[input_cols].copy()
inputs_df


# ### Target column DataFrame

# In[15]:


targets=data_df[" _tempm"]
targets


# In[16]:


inputs_df 


# ### Filling the NAN value with Zero

# In[17]:


inputs_df.fillna(0)


# In[18]:


# Year in the given Dataset
year = pd.to_datetime(inputs_df.datetime_utc).dt.year
year


# ## Segragating the Training set, Validation set and Test Set

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


train_val_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)


# In[21]:


print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)


# In[22]:


train_df.fillna(0);
val_df.fillna(0);
test_df.fillna(0);


# In[23]:


train_inputs = train_df[input_cols].copy().fillna(0)
train_targets = train_df[" _tempm"]


# In[24]:


val_inputs = val_df[input_cols].copy().fillna(0)
val_targets = val_df[" _tempm"]


# In[25]:


test_inputs = test_df[input_cols].copy().fillna(0)
test_targets = test_df[" _tempm"]


# ### Analysing the numeric columns and Categorical columns in the given dataset

# In[26]:


numeric_cols = train_inputs.select_dtypes(include=['int64', 'float64']).columns.tolist()


# In[27]:


categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


# In[28]:


print(list(numeric_cols))


# In[29]:


print(list(categorical_cols))


# In[30]:


missing_counts = train_inputs[numeric_cols].isna().sum().sort_values(ascending=False)
missing_counts[missing_counts > 0]


# ### Statistical Analysis of Train Inputs

# In[31]:


train_inputs.describe()


# ### Encoding Technique for Catogorical column

# In[32]:


from sklearn.preprocessing import OneHotEncoder


# In[33]:


categorical_cols1 = [' _conds',' _wdire']


# In[34]:


inputs_df[categorical_cols1].sample(10)


# In[35]:


inputs_df[categorical_cols1] = inputs_df[categorical_cols1].replace(0, 'NE')


# In[36]:


encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(inputs_df[categorical_cols1])


# In[37]:


encoded_cols = list(encoder.get_feature_names_out(categorical_cols1))


# In[38]:


encoded_cols


# In[39]:


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols1]).copy()
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols1]).copy()
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols1]).copy()


# In[40]:


# Encoded column of Train inputs
train_inputs[encoded_cols]


# In[41]:


X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


# ### Training Data Set

# In[42]:


X_train


# In[43]:


X_test


# ### Importing the Classifier

# In[44]:


from sklearn.tree import DecisionTreeClassifier


# In[45]:


model = DecisionTreeClassifier(random_state=42)


# In[46]:


missing_counts = train_targets.isna().sum()
missing_counts


# ### Imputer for missing values

# In[47]:


from sklearn.impute import SimpleImputer


# In[48]:


imputer = SimpleImputer(strategy = 'constant')


# In[49]:


train_targets1 = pd.DataFrame(train_targets)
val_targets1 = pd.DataFrame(val_targets)
test_targets1 = pd.DataFrame(test_targets)


# In[50]:


targets1 = pd.DataFrame(targets)


# In[51]:


imputer.fit(targets1)


# In[52]:


train_targets1 = imputer.transform(train_targets1)
val_targets1 = imputer.transform(val_targets1)
test_targets1 = imputer.transform(test_targets1)


# In[53]:


from sklearn.neural_network import MLPClassifier


# In[54]:


model1 = MLPClassifier()


# In[55]:


model.fit(X_train, train_targets1)


# In[56]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[57]:


train_preds = model.predict(X_train)


# In[58]:


train_preds


# In[59]:


pd.value_counts(train_preds)


# In[60]:


train_probs = model.predict_proba(X_train)
train_probs


# In[61]:


#accuracy_score(train_targets, train_preds)


# In[62]:


model.score(X_val, val_targets1)


# In[63]:


model = DecisionTreeClassifier(max_depth=2, random_state=42)


# In[64]:


model.fit(X_train, train_targets1)


# In[65]:


model.score(X_train, train_targets1)


# In[66]:


model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, train_targets1)
model.score(X_train, train_targets1)


# In[67]:


model = DecisionTreeClassifier(max_depth=30, random_state=42)
model.fit(X_train, train_targets1)
model.score(X_train, train_targets1)


# In[68]:


model = DecisionTreeClassifier(max_depth=50, random_state=42)
model.fit(X_train, train_targets1)
model.score(X_train, train_targets1)


# In[69]:


model = DecisionTreeClassifier(max_depth=50, random_state=50)
model.fit(X_train, train_targets1)
model.score(X_train, train_targets1)


# In[70]:


model = DecisionTreeClassifier(max_leaf_nodes=128, random_state=42)
model.fit(X_train, train_targets1)
model.score(X_train, train_targets1)


# In[71]:


model = DecisionTreeClassifier(max_leaf_nodes=300, random_state=42)
model.fit(X_train, train_targets1)
model.score(X_train, train_targets1)


# In[72]:


model = DecisionTreeClassifier(max_leaf_nodes=600, random_state=42)
model.fit(X_train, train_targets1)
model.score(X_train, train_targets1)


# In[73]:


model1.fit(X_train, train_targets1)
model1.score(X_train, train_targets1)


# In[74]:


from sklearn.ensemble import RandomForestClassifier


# In[75]:


def test_params(**params):
    model2 = RandomForestClassifier(random_state=42, n_jobs=-1, **params).fit(X_train, train_targets1)
    return model2.score(X_train, train_targets1), model2.score(X_val, val_targets1)


# In[76]:


test_params(max_depth=5)


# In[77]:


test_params(max_leaf_nodes=20)


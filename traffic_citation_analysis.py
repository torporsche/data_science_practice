
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# In[2]:


tickets = pd.read_csv('C:/G/parking_citations.corrupted.csv')


# # Exploratory Data Analysis

# In[3]:


print(len(tickets))


# In[4]:


tickets.head(25)


# In[5]:


###  Get missing values as percentage in all collumns
missing = tickets.isnull().sum()/len(tickets)*100
missing


# In[6]:


### Create a list to drop variables starting with those that have a majority of missing values, too many missing to impute
drop_vars = list(missing[missing>51].index)
drop_vars


# In[7]:


### Find number of unique values in each
levels = tickets.nunique()
levels


# In[8]:


### look at some of the high dimensional features

print(tickets['Latitude'].value_counts().sort_values(ascending=False).head(20))
print('\n')
print(tickets['Location'].value_counts().sort_values(ascending=False).head(20))
print('\n')
print(tickets['Route'].value_counts().sort_values(ascending=False).head(20))


# In[9]:


### Add High Dimensional features to drop list, could potentially use top 100 and create new category
### for all others, not likely to help since most categories distributed fairly evenly

drop_vars.extend(list(levels[levels > 2000].index))


# In[10]:


drop_vars


# In[11]:


set(drop_vars)


# In[12]:


### Look at distrution of states to determine in state/out of state
print(tickets.groupby('RP State Plate')['RP State Plate'].size().sort_values(ascending=False).head(20))


# In[13]:


### Given the variables are mostly categorical, describe is not very benenficial. Checking for sake of completeness
tickets.describe()


# In[14]:


tickets.info()


# In[15]:


len(tickets[tickets.Make.notnull()])


# In[16]:


len(tickets[tickets['Plate Expiry Date'] < 200001])


# # Visulizations

# #  Create Modeling Dataset

# In[17]:


### remove corrupted records, drop_vars list and reset index
tic_mod = tickets[tickets.Make.notnull()]
tic_mod.reset_index(drop=True, inplace=True)
tic_mod.drop(drop_vars, axis=1, inplace = True)
tic_mod.head()


# In[18]:


### Convert date fields to datetime format, Plate Expiry needs to be numeric first due to decimal
tic_mod['Issue Date Fixed'] = pd.to_datetime(tic_mod['Issue Date'], infer_datetime_format = True)
tic_mod['Plate Expiry Date Num'] = pd.to_numeric(tic_mod['Plate Expiry Date']).fillna(0).astype(np.int)


# In[19]:


### In converting Plate Expiry to a date, I discoverd there many invalid dates
### Drop Plate Expiry Dates before 1990 (judgement call) and after 2019 and records with month > 12 or =0

tic_mod = tic_mod[(tic_mod['Plate Expiry Date Num'] < 202001) & (tic_mod['Plate Expiry Date Num'] > 198912) & (tic_mod['Plate Expiry Date Num'] % 100 < 13) & (tic_mod['Plate Expiry Date Num'] % 100 > 0)]


# In[20]:


print(tic_mod['Plate Expiry Date Num'][tic_mod['Plate Expiry Date Num'] > 0].value_counts().sort_values(ascending=False).head())


# In[21]:


### Verify invalid dates have been removed
print(tic_mod['Plate Expiry Date Num'].max())
print(tic_mod['Plate Expiry Date Num'].min())
tic_mod[tic_mod['Plate Expiry Date Num'] % 100 >12]


# In[22]:


### Now that valid dates are remaing, complete converting Plate Expiry to datetime
tic_mod['Plate Expiry Date Fixed'] = pd.to_datetime(tic_mod['Plate Expiry Date Num'], format='%Y%m')


# In[23]:


### Create expired plate flag in case it is predictive and for future analysis
tic_mod['Expired Plates'] = np.where(tic_mod['Issue Date Fixed'] >= tic_mod['Plate Expiry Date Fixed'],'Yes','No')

### Create variable for day of week ticket issued
tic_mod['DOW Ticket Issued'] = tic_mod['Issue Date Fixed'].dt.day_name()
### Create variable for month ticket issued
tic_mod['Month Ticket Issued'] = tic_mod['Issue Date Fixed'].dt.month_name()

### Create residency variable in case it is predictive and for future analysis
tic_mod['Residency'] = tic_mod['RP State Plate'].apply(lambda x: 'In State' if  x == 'CA' else 'Out of State')

tic_mod.head()


# In[24]:


print(tic_mod['Agency'].value_counts().sort_values(ascending=False).head(10))


# In[25]:


### Split time issued into bins, likely more predictive than leaving as continous
def ticket_issued_bins(x):
    if x >= 0 and x <= 400:
        bin =  "Early Morning"
    elif x > 400 and x <= 800:
        bin =  "Morning"
    elif x > 800 and x <= 1159:
        bin = "Late Morning"
    elif x > 1159 and x <= 1600:
        bin = "Afternoon"
    elif x > 1600 and  x <= 2000:
        bin = "Evening"
    else:
        bin = "Late Evening"
    return bin        


# In[26]:


### apply ticket_issued_bin
tic_mod['Issue TOD'] = tic_mod['Issue time'].apply(ticket_issued_bins)


# In[27]:


### Create a Top 25 list based on frequency in data set - needed for target variable
top25 = list(tic_mod['Make'].value_counts().sort_values(ascending=False).head(25).index)
top25


# In[28]:


### Create flag for target variable
tic_mod['Top25'] = np.where(tic_mod.Make.isin(top25), 1,0)
tic_mod.head()


# In[29]:


### See if there is a pattern to color and top25
tic_mod['COUNTER'] = 1

df_agg = tic_mod.groupby(['Top25','Color']).agg({'COUNTER':sum})
g = df_agg['COUNTER'].groupby(level=0, group_keys=False)
res = g.apply(lambda x: x.sort_values(ascending=False))


# In[30]:


res[0].head(10)


# In[31]:


res[1].head(10)


# In[32]:


tic_mod.dtypes


# In[33]:


### remove unneeded fields to simplify dataframe
tic_mod.drop(['Plate Expiry Date Num','Issue Date','Plate Expiry Date', 'Issue time','Make','Issue Date Fixed','Plate Expiry Date Fixed','COUNTER'], axis =1, inplace = True)
tic_mod.head()


# In[34]:


### Convert objects to categories
tic_mod[tic_mod.select_dtypes(['object']).columns] = tic_mod.select_dtypes(['object']).apply(lambda x: x.astype('category'))

### Convert Agency to categorical
tic_mod['Agency'] = tic_mod['Agency'].astype('category')
        
tic_mod.dtypes


# In[35]:


### Check target variable distribution
print(tic_mod['Top25'].value_counts().sort_values(ascending=False))


# In[36]:


### Take a sample of the data set to reduce training due to nature of this exercise
tic_mod = tic_mod.sample(frac=0.04, random_state=321)


# In[37]:


### create one hot encoded df for training
model_df = pd.get_dummies(tic_mod, columns = tic_mod.select_dtypes(['category']).columns)


# In[38]:


print(model_df['Top25'].value_counts().sort_values(ascending=False))


# In[39]:


model_df.head()


# In[41]:


### Determine if any NaN exist, causes issues with RandomUnderSampler
model_df.columns[model_df.isna().any()].tolist()


# In[42]:


### Fix NaN by replacing with 0, could also drop rows
model_df['Fine amount'].fillna(0, inplace=True)


# In[43]:


### Seperate features and target
X = model_df.drop(['Top25'], axis = 1)
y = model_df['Top25']

### Create balanced sample
rus = RandomUnderSampler(random_state=321)
X_resampled, y_resampled = rus.fit_sample(X, y)


# In[44]:


### Verify target class is balanced
unique, counts = np.unique(y_resampled, return_counts=True)
print(np.asarray((unique, counts)).T)


# In[73]:


### Create Training and Test Set
X_train,X_test,y_train,y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=321)
print(len(X_train))
print(len(X_test))


# In[55]:


### Create Basic Tuning Grid

params = {
        'learning_rate': [0.1, 0.01, 0.001],
        'max_depth': [3, 4, 5]
        }


# In[61]:


### Fit Model

folds = 3
iters = 2

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 321)

model = XGBClassifier(n_estimators=100, objective='binary:logistic', nthread=1)

random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=iters, scoring='roc_auc', n_jobs=3, cv=skf.split(X_train,y_train), verbose=3, random_state=321 )

random_search.fit(X_train, y_train)


# In[62]:


print('\n Best hyperparameters:')
print(random_search.best_params_)


# In[69]:


# make predictions for test data
y_pred = random_search.predict(X_test)
y_pred
#predictions = [round(value) for value in y_pred]


# In[74]:


# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# # SQLite and Pandas Analysis

# In[ ]:


tic_mod.head()


# In[ ]:


tic_mod.groupby('DOW Ticket Issued')['DOW Ticket Issued'].size()


# In[ ]:


print(tic_mod.groupby('Top25')['Top25'].size().sort_values(ascending=False))


# In[ ]:


print(tickets.groupby('RP State Plate')['RP State Plate'].size().sort_values(ascending=False).head(10))

print(tickets.groupby('Violation Description')['Violation Description'].size().sort_values(ascending=False))

print(tickets.groupby('Make')['Make'].size().sort_values(ascending=False).head(25))

# tickets.groupby(['RP State Plate']) \
#                              .count() \
#                              .reset_index(name='count') \
#                              .sort_values(['count'], ascending=False) \
#                              .head(5)

#df.groupby(["name"]).apply(lambda x: x.sort_values(["count_1"], ascending = False)).reset_index(drop=True)


# In[ ]:


tickets[tickets['Violation Description'].str.lower().str.contains('tag')==True]


# In[ ]:


tic_mod['']


# In[ ]:


tic_mod.head()


# In[ ]:


tickets[['Residency']][tickets['Violation Description']=='EXPIRED TAGS'].groupby('Residency').size()


# In[ ]:


tickets[['Residency']].groupby('Residency').size()


# In[ ]:


tic_mod.groupby('Expired Plates')['Expired Plates'].size()


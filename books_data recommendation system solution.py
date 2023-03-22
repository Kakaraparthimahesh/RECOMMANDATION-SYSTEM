# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 20:24:00 2022

@author: MAHESH
"""

# REQUIRED LIBRARIES

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distance 
from scipy.spatial.distance import cosine,correlation

# data processing 

Books_data = pd.read_csv("book.csv",encoding='latin1')

Books_data

Books_data.shape
list(Books_data)
Books_data.head()

Books_data.info()
Books_data.isnull().sum()

Books_data.sort_values('User.ID')
Books_data.drop(['Unnamed: 0'],axis=1,inplace=True)

Books_data

# Re-naaming the variable names

Books_data1 = Books_data.rename(columns={'User.ID': 'UserID', 'Book.Title': 'BookTitle', 'Book.Rating': 'BookRating'})
Books_data1
list(Books_data1)

#number of unique users in the dataset

len(Books_data1)
len(Books_data1.UserID.unique()) 

Books_data1['BookRating'].value_counts()
Books_data1['BookRating'].hist()

len(Books_data1.BookTitle.unique())    # unique names 

Books_data1.BookTitle.value_counts()   # count of BookTitle 

user_Books_data2 = Books_data1.pivot_table(index='UserID',columns='BookTitle',values='BookRating')

user_Books_data2              # nan means not the BooksTittles  
user_Books_data2.iloc[0]     
user_Books_data2.iloc[1]
user_Books_data2.iloc[200]    
list(user_Books_data2)

#Impute those NaNs with 0 values

user_Books_data2.fillna(0, inplace=True)     # filling with nal with 0 to rectify the mathenatical error

user_Books_data2.shape

# from scipy.spatial.distance import cosine correlation
# Calculating Cosine Similarity between Users

from sklearn.metrics import pairwise_distances

user_sim = 1 - pairwise_distances( user_Books_data2.values,metric='cosine')

#user_sim = 1 - pairwise_distances( user_df.values,metric='correlation')

user_sim.shape

#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)

#Set the index and column names to user ids 
user_sim_df.index   = Books_data1.UserID.unique()
user_sim_df.columns = Books_data1.UserID.unique()

user_sim_df.iloc[0:5, 0:5]

# Nullifying diagonal values

np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]

#Most Similar Users
user_sim_df.max()       # i want to see the max value 

user_sim_df.idxmax(axis=1)[0:5]

Books_data1[(Books_data1['UserID']==276729) | (Books_data1['UserID']==276729)]

user_1=Books_data1[Books_data1['UserID']==276726]
user_1
user_2=Books_data1[Books_data1['UserID']==276729]
user_2

user_3=Books_data1[Books_data1['UserID']==276736]
user_3
user_4=Books_data1[Books_data1['UserID']==276737]
user_4

user_5=Books_data1[Books_data1['UserID']==276744]
user_5
user_6=Books_data1[Books_data1['UserID']==162107]
user_6

user_7=Books_data1[Books_data1['UserID']==162109]
user_7
user_8=Books_data1[Books_data1['UserID']==162113]
user_8

user_9=Books_data1[Books_data1['UserID']==162121]
user_9
user_10=Books_data1[Books_data1['UserID']==162129]
user_10

pd.merge(user_2,user_4,on='BookTitle',how='inner')
pd.merge(user_2,user_4,on='BookTitle',how='outer')

pd.merge(user_5,user_7,on='BookTitle',how='inner')
pd.merge(user_5,user_7,on='BookTitle',how='outer')

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>><<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# RESULTS AND INFERENCES

# A recommendation system has been build and the persons having similarities
# selected and recommended the books to read by others persons 

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>><<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


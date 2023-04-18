# Now we have two data frames:
# one with GPS data and the other
# with the seismic data features.
# The procedure adopted here was:
# for each GPS station, create a new data frame
# with the 120 columns containing the features
# and the 121th column with the GPS (target) data
# and save it as a csv file

for i in stations:
    df_final = df2.copy()
    df_final.insert(len(df_final.keys()), i,df[i].tolist())
    df_final.to_csv('/Users/Jose/Desktop/{}_df.csv'.format(i))
    
# Now that we have the final matrix (features + target) for each GPS station
# We can proceed with the cleaning process and machine learning algorithm implementation


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# import the csv file just exported
sta1= pd.read_csv('/Users/Jose/Desktop/Big_Data/ALBH_df.csv')

# exclude the columns of stations GHNB, GOBB and SNB
# (deactivated / with lots of missing days)
sta1 = sta1.loc[:,~sta1.columns.str.contains('GHNB')]
sta1 = sta1.loc[:,~sta1.columns.str.contains('GOBB')]
sta1 = sta1.loc[:,~sta1.columns.str.contains('SNB')]

# drop rows that contains NaN elements
# These NaN elements are related with
# missing days from both GPS and/or seismic stations, 
# corrupted files that couldn't be download, etc
sta1 = sta1.dropna(axis = 0)

sta1 = sta1.rename(columns = {'Unnamed: 0': 'Dates'})

#create a list with the name of the features
features_list = list(sta1.keys().tolist())





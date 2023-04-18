# exclude the 'date' from features list
del features_list[0]

# calculate the rolling means for each features
# and append the result (as a new column)
# to the sta1 data frame

for i in features_list:
    sta1['Rolling_{}'.format(i)] = sta1['{}'.format(i)].rolling(60).mean()
# let's keep just the 'rolling' columns    
sta1_rolling = sta1.loc[:,sta1.columns.str.contains('Rolling_')]

# again search for NaN values and exclude them is existent
sta1_rolling = sta1_rolling.dropna(axis = 0)

# Now we need to perform the 
# linear regression with the GPS data
# to get the displacement rate

x_data = np.arange(len(sta1_rolling))
y_data = np.array(sta1_rolling['Rolling_ALBH'])

data = pd.DataFrame(y_data, x_data)
data.insert(0,'x',x_data)

data = data.rename(columns={'x':'x',0:'y'})

regression = np.zeros((len(data.index),2)) #set the regression numpy array empty first
for row in range(0, len(data.index), 1):
    y = data.y[row: row + 61]
    x = data.x[row: row + 61]
    regression[row] = np.polyfit(x, y, 1)

data['alpha'] = regression[:,1]

# as we need the displacement rate date to run the random forest algorith
# let's exclude the columns containing the initial GPS data (previous to rolling linear regression)

sta1_rolling = sta1_rolling.loc[:,~sta1_rolling.columns.str.contains('ALBH')]
# and append the data obtained with the linear regression

sta1_rolling.insert(len(sta1_rolling.keys()),'Disp_rate_ALBH',data['alpha'].tolist())

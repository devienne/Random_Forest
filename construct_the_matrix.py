## CREATING A DATA FRAME

# The strategy used here is the following:
# create a data frame of NaN elements in which the rows
# corresponds to the dates,
# the first 120 columns are the features,
# and the 121th columns is the target
# (i.e., the GPS data)

stat1 = pd.read_csv('/Users/Jose/Desktop/Big_Data/Stations.csv', names = ['stations'])

list_of_station = []

for i in stat1.values.tolist():
    list_of_station.append(i[0])

dates = []

for i in datelist:
    datetime_form = i.date()
    strform = datetime_form.strftime("%Y-%m-%d")
    dates.append(strform)

features = list()

for i in list_of_station: # construct the columns of the data frame
  for j in range(8,13):  
    features.append('kurtosis_'+ str(i) +'_Band{}-{}Hz'.format(j,j+1))
for i in list_of_station:
  for j in range(8,13):
    features.append('var_'+ str(i) +'_Band{}-{}Hz'.format(j,j+1))
for i in list_of_station:
  for j in range(8,13):
    features.append('skew_'+ str(i) +'_Band{}-{}Hz'.format(j,j+1))
for i in list_of_station:
  for j in range(8,13):
    features.append('ValRan_'+ str(i) +'_Band{}-{}Hz'.format(j,j+1))

feat_matrix = np.empty((len(dates),len(features))) 
feat_matrix[:] = np.nan

# feat_matrix is the matrix with NaN element
# The next step is to fill up this matrix with values 
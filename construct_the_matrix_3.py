# Creating the empty data frame for GPS data

for i in os.listdir(): # to get the station's  names
  station_name = os.path.splitext(i)[0]  
  stations.append(station_name)

start = datetime(2010,1,1,0,0,0)
end = datetime(2020,9,30,0,0,0)
datelist = pd.date_range(start, end)

feat_matrix = np.empty((len(datelist),len(stations)))
feat_matrix[:] = np.nan

df = pd.DataFrame(feat_matrix, columns = stations)

df.insert(0,'Datetime',datelist) # insert the column with dates

for i in range(len(df)):
  df['Datetime'][i] = str(df['Datetime'][i]) # tranform the dates into strings
  
 filepath = '/Users/Jose/Desktop/Big_Data/GPS'

 # for each file in the directory
 # open the GPS data, calculate the total horizontal component
 # and fill up this value in the correct position
 # at the GPS data frame

for file in os.listdir():
    df_stat = pd.read_csv('{}/{}'.format(filepath, file), header = 8)
    filename = os.path.splitext(file)[0]
    for i in range(len(df_stat)):
        df_stat['Datetime'][i]  = datetime.strptime(df_stat['Datetime'][i][0:10], '%Y-%m-%d')
        df_stat['Datetime'][i]  = str(df_stat['Datetime'][i])
    df_stat.columns = df_stat.columns.str.replace(' ','_')
  
    df_stat['tot_comp'] = df_stat['_delta_N']+df_stat['_delta_E']

    for i in range(len(df_stat)):
        elem = df_stat['Datetime'][i]
        if elem in df.Datetime.values:
            ind1 = df.loc[df.isin([str(elem)]).any(axis=1)].index.tolist()[0]
            ind2 = df_stat.loc[df_stat.isin([str(elem)]).any(axis=1)].index.tolist()[0]
            df[filename][ind1] = df_stat['tot_comp'][ind2]

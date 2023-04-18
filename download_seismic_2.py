data2 = []

for i in range(len(data)): 
    json_acceptable_string = str(data[i]).replace("'", "\"") 
    data2.append(json_acceptable_string) 

for i in range(len(data2)):
    data2[i] = json.loads(data2[i])

df = pd.DataFrame(data2)
df = df.dropna(subset=['code'])
df = df.reset_index(drop=True)

start = datetime.datetime(2010, 1, 1)

end = datetime.datetime(2020, 9, 30)

# In column 'endDate' all stations with non NaN
# have been set out of order before 30/09/2020
# therefore they should not be included

for i in range(len(df)):
    df['startDate'][i] = datetime.datetime.strptime(df['startDate'][i][0:10], '%Y-%m-%d')  
    if df['endDate'][i] == np.nan:
        pass
    elif isinstance(df['endDate'][i],str):
        df['endDate'][i] = datetime.datetime.strptime(df['endDate'][i][0:10], '%Y-%m-%d') 

df = df.query('endDate != endDate')
df = df.reset_index(drop=True)

df_final = df.loc[(df.startDate < start)]
df_final = df_final.reset_index(drop=True)
# df_final contains the name of the station within the radius of 0.6o from 48.9o N, -123.8o W 

stations = list(df_final['code'])

with open('/Users/Jose/Desktop/Big_Data/Stations.csv', 'w') as f:
    wr = csv.writer(f, delimiter="\n")
    wr.writerow(stations)
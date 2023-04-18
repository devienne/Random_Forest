#### DOWNLOAD GPS DATA ####

gps_ser = 'https://web-services.unavco.org/gps/metadata/sites/v1?'
minlat = 'minlatitude=47.9&'
maxlat = 'maxlatitude=49.9&'
minlong= 'minlongitude=-125.4&'
maxlong = 'maxlongitude=-123.4&'
rest = 'starttime=&endtime=&summary=false'

url = gps_ser + minlat + maxlat + minlong + maxlong + rest

response = requests.get(url,headers={'accept': 'application/json'}) 

data = response.json()

station_names=[]

for i in range(len(data)):
    ls = list(data[i].values())
    station_names.append(ls[0])

station_names=sorted(set(station_names))

print(station_names)


for i in range(len(station_names)):
    part1 = 'https://web-services.unavco.org/gps/data/position/{}'.format(station_names[i])
    part2 = '/v3?analysisCenter=cwu&referenceFrame=nam14&'
    part3 = 'starttime=&endtime=&report=short&dataPostProcessing=Uncleaned&'
    part4 = 'refCoordOption=from_analysis_center'
    url = part1 + part2 + part3 + part4
    response_station = requests.get(url)
    export = '/Users/Jose/Desktop/Big_Data/GPS/{}.csv'.format(station_names[i])
    with open(export,'w') as f:
        writer = csv.writer(f)
        for line in response_station.iter_lines():
            writer.writerow(line.decode('utf-8').split(','))


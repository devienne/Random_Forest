# Now that we selected the station 
# we can download the raw seismic data from them 

from datetime import datetime

start = datetime.datetime(2010,1,1,0,0)
end = datetime.datetime(2020,9,30,0,0)

datelist = pd.date_range(start, end).tolist()

# Generating the url's

for station in stations:
    for i in datelist:
        year = i.date().year
        month = i.date().month
        if month<10:
            month ='0{}'.format(month)
            day = i.date().day
        if day<10:
            day = '0{}'.format(day)

        url1 ='http://service.iris.edu/fdsnws/dataselect/1/query?'
        station = 'sta={}&'.format(station)
        starttime = 'starttime={}-{}-{}T00:00:00&'.format(year, month, day)
        enddate = 'endtime={}-{}-{}T01:00:00&'.format(year, month, day)         
        end = 'format=miniseed&nodata=404'
        
        url = url1 + station + starttime + endtime + end
        filename = '{}_{}-{}-{}'.format(station,year,month,day)
   
        if os.path.exists('/Users/Jose/Desktop/Big_Data/Seismic_2010_2010/{}.mseed'.format(filename)):
            pass
        else:
            response = requests.get(url,allow_redirects = True)
            if response.text[0:5] == 'Error':
                pass
            else:
                export = '/Users/Jose/Desktop/Big_Data/Seismic_2020_2010/{}.mseed'.format(filename)
                open(export,'wb').write(response.content)

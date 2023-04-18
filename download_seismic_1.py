# DOWNLOADING SEISMIC DATA

import requests # for r requesting data from urls using Pyhton
import json # to read/manipulate json files
from lxml import objectify # to turn xml file content into Python objects                         
from io import BytesIO # also to turn xml file content into Python objects                              
import pandas as pd # to represent and manipulate data as data frames
import numpy as np # the classic numpy
import csv # to save .csv files from GPS stations
import os # to interact with directories
import datetime # to work with dates

url_1 = 'http://service.iris.edu/fdsnws/station/1/query?cha=HHE&starttime=2010-01-01'
url_2 = '&endtime=2020-09-30&level=station&format=xml&lat=48.900&lon=-123.900&'
url_3 = 'minradius=0.0&maxradius=0.600&includecomments=true&nodata=404'

url = url_1 + url_2 + url_3

response = requests.get(url,headers={'accepted':'application/xml'})

ob = objectify.parse(BytesIO(response.content))

root = ob.getroot()

data = [] 

for i in root.getchildren():
    for j in i.getchildren():
        data.append(j.attrib) 

data = list(filter(None,data))



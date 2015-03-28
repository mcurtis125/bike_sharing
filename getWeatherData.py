#!/usr/bin/python

import urllib
import datetime

APIKEY = '54deb0d6967e0fcb6d965bee2cc94ba4'

now = datetime.datetime.now()

path = "/Users/stavy92/Documents/DP_local/WeatherData/"

# out filename for each city
outfilenameMtl = now.strftime(path + "Montreal/%Y_%m_%d__%H_%M_%S.json")
outfilenameTor = now.strftime(path + "Toronto/%Y_%m_%d__%H_%M_%S.json")
outfilenameOtt = now.strftime(path + "Ottawa/%Y_%m_%d__%H_%M_%S.json")
outfilenameWas = now.strftime(path + "Washington/%Y_%m_%d__%H_%M_%S.json")
outfilenameBos = now.strftime(path + "Boston/%Y_%m_%d__%H_%M_%S.json")
outfilenameChi = now.strftime(path + "Chicago/%Y_%m_%d__%H_%M_%S.json")
outfilenameNyc = now.strftime(path + "NYC/%Y_%m_%d__%H_%M_%S.json")

# in weather data for each city
datafileMtl = urllib.urlopen('https://api.forecast.io/forecast/' + APIKEY + '/45.5017,-73.5673?exclude=minutely,hourly,daily,alerts,flags')
datafileTor = urllib.urlopen('https://api.forecast.io/forecast/' + APIKEY + '/43.7,-79.4?exclude=minutely,hourly,daily,alerts,flags')
datafileOtt = urllib.urlopen('https://api.forecast.io/forecast/' + APIKEY + '/45.4214,-75.6919?exclude=minutely,hourly,daily,alerts,flags')
datafileWas = urllib.urlopen('https://api.forecast.io/forecast/' + APIKEY + '/38.9047,-77.0164?exclude=minutely,hourly,daily,alerts,flags')
datafileBos = urllib.urlopen('https://api.forecast.io/forecast/' + APIKEY + '/42.3601,-71.0589?exclude=minutely,hourly,daily,alerts,flags')
datafileChi = urllib.urlopen('https://api.forecast.io/forecast/' + APIKEY + '/41.8369,-87.6847?exclude=minutely,hourly,daily,alerts,flags')
datafileNyc = urllib.urlopen('https://api.forecast.io/forecast/' + APIKEY + '/40.7127,-74.0059?exclude=minutely,hourly,daily,alerts,flags')

# write data to files
 
# mtl
fout = open(outfilenameMtl, 'w')
fout.write(datafileMtl.read())
datafileMtl.close()
fout.close()

# tor
fout = open(outfilenameTor, 'w')
fout.write(datafileTor.read())
datafileTor.close()
fout.close()

# ott
fout = open(outfilenameOtt, 'w')
fout.write(datafileOtt.read())
datafileOtt.close()
fout.close()

# was
fout = open(outfilenameWas, 'w')
fout.write(datafileWas.read())
datafileWas.close()
fout.close()

# bos
fout = open(outfilenameBos, 'w')
fout.write(datafileBos.read())
datafileBos.close()
fout.close()

# chi
fout = open(outfilenameChi, 'w')
fout.write(datafileChi.read())
datafileChi.close()
fout.close()

# nyc
fout = open(outfilenameNyc, 'w')
fout.write(datafileNyc.read())
datafileNyc.close()
fout.close()
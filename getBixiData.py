#!/usr/bin/python

import urllib
import datetime

now = datetime.datetime.now()

path = "/Users/stavy92/Documents/DP_local/BixiData/"

# out filename for each city
outfilenameMtl = now.strftime(path + "Montreal/%Y_%m_%d__%H_%M_%S.xml")
outfilenameTor = now.strftime(path + "Toronto/%Y_%m_%d__%H_%M_%S.xml")
#outfilenameOtt = now.strftime(path + "BixiData/Ottawa/%Y_%m_%d__%H_%M_%S.xml")
outfilenameWas = now.strftime(path + "Washington/%Y_%m_%d__%H_%M_%S.xml")
outfilenameBos = now.strftime(path + "Boston/%Y_%m_%d__%H_%M_%S.xml")
outfilenameChi = now.strftime(path + "Chicago/%Y_%m_%d__%H_%M_%S.json")
outfilenameNyc = now.strftime(path + "NYC/%Y_%m_%d__%H_%M_%S.json")

# in bike station data for each city
datafileMtl = urllib.urlopen('http://montreal.bixi.com/data/bikeStations.xml')
datafileTor = urllib.urlopen('http://www.bikesharetoronto.com/data/stations/bikeStations.xml')
#datafileOtt = urllib.urlopen('http://capital.bixi.com/data/bikeStations.xml')
datafileWas = urllib.urlopen('http://www.capitalbikeshare.com/data/stations/bikeStations.xml')
datafileBos = urllib.urlopen('http://www.thehubway.com/data/stations/bikeStations.xml')
datafileChi = urllib.urlopen('http://www.divvybikes.com/stations/json')
datafileNyc = urllib.urlopen('http://www.citibikenyc.com/stations/json')

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
#fout = open(outfilenameOtt, 'w')
#fout.write(datafileOtt.read())
#datafileOtt.close()
#fout.close()

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
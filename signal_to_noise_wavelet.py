# -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt
import pywt
import sys
import numpy as np 
import pandas as pd
from pandas import DataFrame 
import argparse
import pdb


def File_Writer(file_name, data1, data2):
    
    file = open(str(file_name) + '_wavelet.csv','a')
    
    #file.write('# ' + str(file_name) + '_wavelet_procesing'+  '\n')
    #file.write('# intensity   counts \n')
    
    for h in range(len(data1)):
        file.write(str(data1[h]) + ' ' + str(data2[h])+ '\n')
    file.close()


ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', help = 'path to the image')
args = vars(ap.parse_args())  
file = args['file']


t = []
f = []
with open(file, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter= ' ', quotechar = '#')
    
    for row in reader:
        try:
            t.append(float(row[0]))
            f.append(float(row[1]))
        except ValueError:
            pass

index = np.asarray(t)
data = np.asarray(f)

#pdb.set_trace()

# Get data
#file_object  = open('19A-1400V-Z4.csv', 'r')
#raw_data = file_object.readlines()
#index = []
#data = []
#for i in range(len(raw_data)-1):
#    x, y = raw_data[i].split('\n')
#    X = float(x)
#    Y = float(y)
#    index.append(X)
#    data.append(Y)

# Create wavelet object and define parameters
w = pywt.Wavelet('sym4')
maxlev = pywt.dwt_max_level(len(data), w.dec_len)
print("maximum level is " + str(maxlev))
threshold = 0.35 # Threshold for filtering can be computed http://gwyddion.net/documentation/user-guide-en/wavelet-transform.html

# Decompose into wavelet components, to the level selected

coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
plt.figure()
for i in range(1, len(coeffs)):
    plt.subplot(maxlev, 1, i)
    plt.plot(coeffs[i])
    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
    plt.plot(coeffs[i])


datarec = pywt.waverec(coeffs, 'sym4')
indexrec = np.arange(0, len(datarec), 1)*2

mintime = 9 # Start point in the time scale
maxtime = mintime + 2000 # The added time is in ms


File_Writer(file, indexrec, datarec)

#pdb.set_trace()

plt.figure()
plt.subplot(2, 1, 1)
#plt.plot(index[mintime:maxtime], data[mintime:maxtime])
plt.plot(index, data)
plt.xlabel('time (s)')
plt.ylabel('Frequency (Hz)')
plt.title("Raw signal")
plt.subplot(2, 1, 2)
#plt.plot(index[mintime:maxtime], datarec[mintime:maxtime])
plt.plot(indexrec, datarec)
plt.xlabel('time (s)')
plt.ylabel('Frequency (Hz)')
plt.title("De-noised signal")

plt.tight_layout()
plt.show()

# fs = 500
# f, t, Sxx = signal.spectrogram(data, fs)
# plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.ylim(0, 50)
# plt.show()

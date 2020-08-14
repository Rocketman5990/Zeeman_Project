import matplotlib.pyplot as plt
import pywt
import sys
import numpy as np 
import pandas as pd
from pandas import DataFrame, read_csv

# Get data
file_object  = open('TestElectro2.txt', 'r')
raw_data = file_object.readlines()
index = []
data = []
for i in range(len(raw_data)-1):
    x, y = raw_data[i].split('\t')
    X = float(x)
    Y = float(y)
    index.append(X)
    data.append(Y)

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

mintime = 9 # Start point in the time scale
maxtime = mintime + 2000 # The added time is in ms

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(index[mintime:maxtime], data[mintime:maxtime])
plt.xlabel('time (s)')
plt.ylabel('Volts (mV)')
plt.title("Raw signal")
plt.subplot(2, 1, 2)
plt.plot(index[mintime:maxtime], datarec[mintime:maxtime])
plt.xlabel('time (s)')
plt.ylabel('Volts (mV)')
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

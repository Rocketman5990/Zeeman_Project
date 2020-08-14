# -*- coding: utf-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.ticker as tkr
from matplotlib.ticker import AutoMinorLocator
from lmfit import Model
import sys
import argparse
from scipy.signal import argrelextrema

from matplotlib import style
style.use('fast')

import pdb


def lineal(x, m, b):
    y = m*x + b
    return y

def sinusoid(x, A, w, p):
    y = A * np.sin(w * x + p)
    return y


t = []
f = []


#LECTURA DE DATOS----------------------------------------------------------------------------------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', help = 'path to the image')
args = vars(ap.parse_args())  
file = args['file']

with open(file, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter= ' ', quotechar = '#')
    
    for row in reader:
        try:
            t.append(float(row[0]))
            f.append(float(row[1]))
        except ValueError:
            pass

t = np.asarray(t)
f = np.asarray(f)
#pdb.set_trace()
#AJUSTE LINEAL------------------------------------------------------------------------------------------------------------------------------------------------------

mod1 = Model(lineal)
pars1 = mod1.make_params(m=1, b=1)
result1 = mod1.fit(f, pars1, x=t)
#sig = 8
params1 = result1.best_values
m = params1["m"]
b = params1["b"]
#dely1 = result1.eval_uncertainty(sigma=sig)
ylim_fit = lineal(t, m, b)

lin_substract = np.zeros(len(f))
#Substaraccion de ajuste
for k in range(len(f)):
    lin_substract[k] = f[k] - ylim_fit[k]
    
#pdb.set_trace()   

#AJUSTE SENO-------------------------------------------------------------------------------------------------------------------------------------------------------
#mod2 = Model(sinusoid)
#pars2 = mod2.make_params(A = 0.25, w=800, p=1)
#result2 = mod2.fit(ylim_fit, pars2, x=t)
#params2 = result2.best_values
#A = params2["A"]
#w = params2["w"]
#p = params2["p"]
#ysen_fit = sinusoid(t, A, w, p)

#sen_substract = np.zeros(len(lim_substract))
#for k in range(len(lim_substract)):
#    sen_substract[k] = lim_substract[k] - ysen_fit[k]


#pdb.set_trace()

#SELECCION DE MINIMOS Y MAXIMOS DATOS ORIGINALES------------------------------------------------------------------------------------------------------------------------------------

minindex, = argrelextrema(f, np.less)
maxindex, = argrelextrema(f, np.greater)

t_minsl = np.zeros(len(minindex))
f_minsl = np.zeros(len(minindex))

t_maxsl = np.zeros(len(maxindex))
f_maxsl = np.zeros(len(maxindex))



for i in range(len(minindex)):
    t_minsl[i] = t[minindex[i]]
    f_minsl[i] = f[minindex[i]]

for j in range(len(maxindex)):
    t_maxsl[j] = t[maxindex[j]]
    f_maxsl[j] = f[maxindex[j]]

#pdb.set_trace()

#SELECCION DE MINIMOS Y MAXIMOS EN EXTRACCION------------------------------------------------------------------------------------------------------------------------------------
linsubs_minindex, = argrelextrema(lin_substract, np.less)
linsubs_maxindex, = argrelextrema(lin_substract, np.greater)

len1 = len(linsubs_minindex)
lt_minsl = np.zeros(len(linsubs_minindex))
lsubs_minsl = np.zeros(len1)

len2 = len(linsubs_maxindex)
lt_maxsl = np.zeros(len(linsubs_maxindex))
lsubs_maxsl = np.zeros(len2)

for k in range(len(linsubs_minindex)):
    lt_minsl[k] = t[linsubs_minindex[k]]
    lsubs_minsl[k] = lin_substract[linsubs_minindex[k]]
    
for k in range(len(linsubs_maxindex)):
    lt_maxsl[k] = t[linsubs_maxindex[k]]
    lsubs_maxsl[k] = lin_substract[linsubs_maxindex[k]]

#pdb.set_trace()

#CALCULO DE DIFERENCIA---------------------------------------------------------------------------------------------------------------------------------------------------


if len1 > len2:
    factor1 = np.absolute(lsubs_minsl[0:len2])
    factor2 = np.absolute(lsubs_maxsl[0:len2])
    noise = np.absolute(factor1 - factor2)    

if len2 > len1:
    factor1 = np.absolute(lsubs_minsl[0:len1])
    factor2 = np.absolute(lsubs_maxsl[0:len1])
    noise = np.absolute(factor1 - factor2)

if len2 == len1:
    factor1 = np.absolute(lsubs_minsl[0:len1])
    factor2 = np.absolute(lsubs_maxsl[0:len1])
    noise = np.absolute(factor1 - factor2)

#pdb.set_trace()
mean_noise = np.mean(noise)    
#pdb.set_trace()

#CALCULO DE MAXIMOS---------------------------------------------------------------------------------------------------------------------------------------------------------

min_data = min(lin_substract)
max_data = max(lin_substract)
diff_data = max_data - min_data

#SIGNAL TO NOISE-----------------------------------------------------------------------------------------------------------------------------------------------------------

if diff_data > mean_noise: 
    sig_to_noi = mean_noise/diff_data

if diff_data < mean_noise: 
    sig_to_noi = diff_data/mean_noise
    
if diff_data == mean_noise: 
    sig_to_noi = diff_data/mean_noise

if np.isnan(mean_noise) == True: 
    sig_to_noi = 'nan'
    
if np.isnan(diff_data) == True: 
    sig_to_noi = 'nan'    

#pdb.set_trace()
#GRAFICACION LINEAL--------------------------------------------------------------------------------------------------------------------------------------------------------
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.plot(t, f, '-', color ='b', linewidth=2)
ax1.plot(t, ylim_fit, '-', color ='r', linewidth=2)

ax1.plot(t_minsl, f_minsl, 'o', color ='g', linewidth=2)
ax1.plot(t_maxsl, f_maxsl, 'o', color ='r', linewidth=2)
ax1.set_title('Data and fit')
ax1.set_xlim(t[0],t[-1])
plt.xticks(np.arange(t[0],t[-1], step=100), fontsize = 25)
ax1.set_xlabel("Time (s)", fontsize=30)

ax1.set_ylabel("Frequency (Hz)", fontsize=30)
yy1, locs = plt.yticks()
ll1 = ['%.0f' % a for a in yy1]#Esto es para que los numero en la grafica salgan completos
plt.yticks(yy1, ll1, fontsize=25)

ax1.spines['right'].set_linewidth(4)
ax1.spines['left'].set_linewidth(4)
ax1.spines['top'].set_linewidth(4)
ax1.spines['bottom'].set_linewidth(4)

ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(which='both', width=4, direction='in')
ax1.tick_params(which='major', length=15, bottom=True, top=True, left=True, right=True)
ax1.tick_params(which='minor', length=8, bottom=True, top=True, left=True, right=True)
ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

#GRAFICACION SENOSOIDAL-------------------------------------------------------------------------------------------------------------------------------------------------
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_title('fit substraction')
ax2.plot(t, lin_substract, '-', color ='c', linewidth=2)
ax2.plot(lt_minsl, lsubs_minsl, 'o', color ='g', linewidth=2)
ax2.plot(lt_maxsl, lsubs_maxsl, 'o', color ='r', linewidth=1)


ax2.set_xlim(t[0],t[-1])
plt.xticks(np.arange(t[0],t[-1], step=100), fontsize = 11)
ax2.set_xlabel("Time (s)", fontsize=11)

ax2.set_ylabel("Frequency (Hz)", fontsize=11)
#yy2, locs = plt.yticks()
#ll2 = ['%000.00000f' % a for a in yy2]#Esto es para que los numero en la grafica salgan completos
#plt.yticks(yy2, ll2, fontsize=25)

ax2.spines['right'].set_linewidth(4)
ax2.spines['left'].set_linewidth(4)
ax2.spines['top'].set_linewidth(4)
ax2.spines['bottom'].set_linewidth(4)

ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(which='both', width=4, direction='in')
ax2.tick_params(which='major', length=15, bottom=True, top=True, left=True, right=True)
ax2.tick_params(which='minor', length=8, bottom=True, top=True, left=True, right=True)
ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)


print('\n')
print(f'mean_noise: {mean_noise} \n')
print(f'signal max value: {max_data} \n')
print(f'signal min value: {min_data} \n')

if diff_data > mean_noise:
    print(f'signal to noise: mean_noise/diff_data \n')
    print(f'signal to noise: {sig_to_noi} \n')


if diff_data < mean_noise: 
    print(f'signal to noise: diff_data/mean_noise \n')
    print(f'signal to noise: {sig_to_noi} \n')

if diff_data == mean_noise: 
    print(f'signal to noise: diff_data/mean_noise \n')
    print(f'signal to noise: {sig_to_noi} \n')

if np.isnan(diff_data) == True: 
    print(f'signal to noise: nan \n')

if np.isnan(mean_noise) == True: 
    print(f'signal to noise: nan \n')
    
    
plt.show()
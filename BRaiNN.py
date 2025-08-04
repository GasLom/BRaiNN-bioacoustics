##############################################
#BRaiNN_bioacoustics EI 2025.
##############################################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

#import sys, os, os.path
#from os import sys
import shutil
import glob
import re

from datetime import datetime
import psutil 
import resource 

start_time = datetime.now()
process = psutil.Process() 


def sounds_processor2(wav_list, threshold, min_f, max_f):
    #threshold = 0.1
    peaks_f = []
    ID = []
    count = 0
    for wav in wav_list:
        s_rate, signal = wavfile.read(wav)
        N = len(signal)  # Number of sample points
        T = 1.0 / s_rate # Sample period 
        yf = fft(signal)
        tf = fftfreq(N, T)
        yf2 = np.abs(yf[0:N//2])**2 # power
        tf2 = tf[0:N//2]*1.e-3   # kHz
        
        maxval_s = max(yf2)
        TH_s = threshold*maxval_s
        peaks_s, _ = find_peaks(yf2, height=TH_s)
        peaks_f_values = tf2[peaks_s]
        
        peaks_f_values = np.delete(peaks_f_values, np.where(peaks_f_values < min_f))
        peaks_f_values = np.delete(peaks_f_values, np.where(peaks_f_values > max_f))
        
        
        ####RETURN 'SILENCE' IF NO PEAK FREQUENCIES (ACROSS WHOLE FILE) IN RANGE
        
        if (peaks_f_values).size == 0:
            shutil.move(wav, "SILENCE")
            continue
        
        peaks_inrange = np.where((tf2 > min_f) & (tf2 < max_f))

        yf3 = yf2[peaks_inrange]
        tf3 = tf2[peaks_inrange]
        peaks, _ = find_peaks(yf3, height=0)               
        maxval = max(yf3[peaks], default=0)
        TH = threshold*maxval
        peaks, _ = find_peaks(yf3, height=TH)
        peaks_f_values = tf3[peaks]
        
        peaks_f_values = np.delete(peaks_f_values, np.where(peaks_f_values < min_f))
        peaks_f_values = np.delete(peaks_f_values, np.where(peaks_f_values > max_f))
        
        ### Remove unclassifiable signals (Comment out this if statement for Model 1 results)
        
        if len(peaks) > 0:
            max_peak = peaks[np.argmax(yf3[peaks])]
            max_peak_f = tf3[max_peak]
            if 49 <= max_peak_f <= 51:
                shutil.move(wav, "UNCLASS")
                continue 
        
        peaks_f.append(peaks_f_values)        
        ID_in = np.zeros((len(peaks_f_values)))+count
        ID.append(ID_in)        
        count += 1    
        
    cat = np.array([np.concatenate((peaks_f))]).T
    ID = np.array([np.concatenate((ID))]).T
    cat = np.concatenate((ID, cat), axis=1)
    return cat


def neuron_firing(n_signals, peaks_f, freqrange):
    neurons = np.zeros((n_signals, len(freqrange)))-1

    for j in range(n_signals):
        peaks_f1 = peaks_f[peaks_f[:,0] == j, 1]
        index = np.zeros(len(peaks_f1))
        
        for i in range(len(peaks_f1)):
            dist = np.abs(freqrange-peaks_f1[i])
            index[i] = dist.argmin()

        for i in index:
            neurons[j, int(i)] = 1
            
    return neurons


def network_model(peaks_f):    
    lf = 45
    hf = 60
    binwidth = 0.9
    freqrange = np.arange(lf,hf+binwidth,binwidth)
    maxID = int(max(peaks_f[:,0]))
    neurons = neuron_firing(maxID+1, peaks_f, freqrange)
       
    return neurons, binwidth, lf, hf


def power_spec(wav_list, threshold):
    global lf, hf, binwidth

    fig, ax = plt.subplots(1,1, figsize=(7,5))

    for wav in wav_list:
        
        match = re.search("PIPI_", wav)
        if match:
            wavname = "Common (PIPI)"
            
        match = re.search("PIPY_", wav)
        if match:
            wavname = "Soprano (PIPY)"
            
        s_rate, signal = wavfile.read(wav)
        N = len(signal)
        T = 1.0 / s_rate     
        yf = fft(signal)
        tf = fftfreq(N, T)
        yf2 = np.abs(yf[0:N//2])**2
        tf2 = tf[0:N//2]*1.e-3
        
        peaks_inrange = np.where((tf2 > lf) & (tf2 < hf))

        yf3 = yf2[peaks_inrange]
        tf3 = tf2[peaks_inrange]
        peaks, _ = find_peaks(yf3, height=0)
        
        maxval = max(yf3[peaks])
        TH = threshold*maxval
        peaks, _ = find_peaks(yf3, height=TH)

        bins=np.arange((lf),(hf+1), step=binwidth)
        bins=np.around(bins, decimals=2, out=None)

        ax.plot(tf2,yf2/maxval, label=wavname) #relative power vs frequency
        ax.plot(tf3[peaks], yf3[peaks]/maxval, 'x', color='black')
        ax.set_xlim(lf,hf)
        ax.set_ylim(-0.1,1.1)
        ax.set_xticks(bins)
        ax.set_xticklabels(bins, rotation=45)
        ax.grid(axis = 'x', color = '0.80')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Relative Power')           

    ax.legend()
    fig.tight_layout(pad=0.1)
    fig.savefig('HNN_stored_sounds.png', dpi=600)
    return


def energy(W,x):        
    E=0
    for i in range(len(x)):
        for j in range(len(x)):
            if j != i:
                E += -1/2*W[i,j]*x[i]*x[j]        
    return E


# More efficient but identical energy calculation
def energy2(W,x):
    E = np.dot(W,x)
    E = np.dot(x.T,E)
    return -0.5*np.diagonal(E)


def count(wav, count1, count2, count3):   
    match = re.search("/PIPI_", wav)
    if match:
        count1 += 1
    match = re.search("/PIPY_", wav)
    if match:
        count2 += 1
    match = re.search("/S-", wav)
    if match:
        count3 += 1
    return count1, count2, count3


#################################################################
#Hopfield Network Model - Storing signals into the network memory
#################################################################

##Assigning the number of neurons and rows and columns for the matrix.

wav_list = glob.glob("stored_sounds/*.wav")

threshold = 0.1
min_f = 45
max_f = 60
peaks_f = sounds_processor2(wav_list, threshold, min_f, max_f)
neurons, binwidth, lf, hf = network_model(peaks_f)
freqrange = np.arange(lf,hf+binwidth,binwidth)

##Index neurons by species file label
neurons = pd.DataFrame(neurons)
labels = ['']*len(wav_list)

for i,wav in zip(range(len(wav_list)),wav_list):
    labels[i] = wav[14:18]
neurons.index = labels

##Change col numbers to match frequencies/neurons
neurons.columns = [x for x in freqrange]
pd.set_option('display.max_columns', None)

sf = neurons.iloc[:,:].values
N = len(sf[0])
sf = np.asmatrix(sf)
row_count = len(sf)

W = np.zeros((N,N))
for i in range(row_count):    
    p = sf[i]
    p = p.reshape(N,1)
    W += p*p.T-np.identity(N)

end_time_train = datetime.now()
cpu_usage = resource.getrusage(resource.RUSAGE_SELF).ru_utime
memory_usage = process.memory_info().rss  # in bytes 
print('Training:')
print('Time elapsed:', end_time_train - start_time, 'hours:minutes:seconds')
print(f"CPU Time Usage: {cpu_usage} seconds") 
print(f"Memory Usage: {memory_usage / (1024 ** 2):.2f} MB")


#####################################################     
##Identifying/Classifying Signals and Model Metrics
#####################################################

##Firstly identify and remove SILENCES and UNCLASSIFIABLES for Model 2
wav_input = glob.glob("DATA/*.wav")
num_files = len(wav_input)

input_total = len(wav_input)
pipi_count = 0
pipy_count = 0
s_count = 0
for wav in wav_input:         
    pipi_count, pipy_count, s_count = count(wav, pipi_count, pipy_count, s_count)    

peaks_f_input = sounds_processor2(wav_input, threshold, min_f, max_f)

##Process and Classify without SILENCES and UNCLASSIFIABLES for Model 2
wav_input = glob.glob("DATA/*.wav")
new_num_files = len(wav_input)
pi_count_new = 0
py_count_new = 0
s_count_new = 0
for wav in wav_input:    
    pi_count_new, py_count_new, s_count_new = count(wav, pi_count_new, py_count_new, s_count_new)

peaks_f_input = sounds_processor2(wav_input, threshold, min_f, max_f)

n_signals = len(wav_input)
input_signal = neuron_firing(n_signals, peaks_f_input, freqrange)
x = np.asmatrix(input_signal)
x = x.T

##To examine initial energy:
#print('Initial')
#print(x.T)
#print('Energy:')
#print(energy(W,x))
#print(energy2(W,x))

z = np.zeros(N)
for k in range(4):
    z = np.dot(W, x)
    x = np.sign(z)
    ##To examine energy after each iteration:
    #print('Updated {}'.format(k+1))
    #print(x.T)
    #print('Energy:')
    #print(energy(W,x))
    #print(energy2(W,x))

##Sort files by species
pipi = neurons.iloc[0,:].values
pipy = neurons.iloc[1,:].values

count_pi_in_pi = 0
count_py_in_pi = 0
count_s_in_pi = 0
count_py_in_py = 0
count_pi_in_py = 0
count_s_in_py = 0
count_pi_in_un = 0
count_py_in_un = 0
count_s_in_un = 0


##################################################
##Compare final state of network to stored signals
##################################################


for i, wav in zip(range(n_signals), wav_input):
    if (x[:,i].T == pipi).all():
        count_pi_in_pi, count_py_in_pi, count_s_in_pi = count(wav, count_pi_in_pi, count_py_in_pi, count_s_in_pi)
        shutil.move(wav, "PIPI")

    elif (x[:,i].T == pipy).all():
        count_pi_in_py, count_py_in_py, count_s_in_py = count(wav, count_pi_in_py, count_py_in_py, count_s_in_py)
        shutil.move(wav, "PIPY")
    
    else:
        count_pi_in_un, count_py_in_un, count_s_in_un = count(wav, count_pi_in_un, count_py_in_un, count_s_in_un)
        shutil.move(wav, "UNID")

print('Support = ', new_num_files)
print('\nPIPI Support = ', pi_count_new, 'PIPY SUPPORT = ', py_count_new)

print('\nPi in Pi = ', count_pi_in_pi, '  Pi in Py = ', count_pi_in_py, '  Pi in Un = ', count_pi_in_un)
print(  'Py in Pi = ', count_py_in_pi, '   Py in Py = ', count_py_in_py, '  Py in Un = ', count_py_in_un)

pi_precision = count_pi_in_pi/(count_pi_in_pi + count_py_in_pi + count_s_in_pi)
pi_recall = count_pi_in_pi/(count_pi_in_pi + count_pi_in_py + count_pi_in_un)

py_precision = count_py_in_py/(count_py_in_py + count_pi_in_py + count_s_in_pi)
py_recall = count_py_in_py/(count_py_in_py + count_py_in_pi + count_py_in_un)

F1_pi = 2 * ((pi_precision*pi_recall)/(pi_precision+pi_recall))
F1_py = 2 * ((py_precision*py_recall)/(py_precision+py_recall))

print('\nPIPI precision = ', round(pi_precision, 2), '  recall = ', round(pi_recall, 2),
      '  F1 = ', round(F1_pi, 2))
print('PIPY precision = ', round(py_precision, 2), '  recall = ', round(py_recall, 2),
      '  F1 = ', round(F1_py, 2))

end_time = datetime.now()
cpu_usage = resource.getrusage(resource.RUSAGE_SELF).ru_utime
memory_usage = process.memory_info().rss  # in bytes 
print('Full Execution (Training and Classification)')
print('Execution time:', end_time - start_time)
print(f"CPU Time Usage: {cpu_usage} seconds") 
print(f"Memory Usage: {memory_usage / (1024 ** 2):.2f} MB")  # Convert bytes to MB 


##################################################
#BRaiN_bioacoustics EI 2025. Analyse/Plot Results.
##################################################


import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import glob


def power_spec(wav, threshold):
    global min_f, max_f, binwidth

    s_rate, signal = wavfile.read(wav)
    N = len(signal)
    T = 1.0 / s_rate     
    yf = fft(signal)
    tf = fftfreq(N, T)
    yf2 = np.abs(yf[0:N//2])**2
    tf2 = tf[0:N//2]*1.e-3
        
    peaks_inrange = np.where((tf2 > min_f) & (tf2 < max_f))

    yf3 = yf2[peaks_inrange]
    tf3 = tf2[peaks_inrange]
    peaks, _ = find_peaks(yf3, height=0)       
    maxval = max(yf3[peaks], default = 0)        
    TH = threshold*maxval        
    peaks, _ = find_peaks(yf3, height=TH)

    return tf2, yf2, tf3, yf3, peaks, maxval


#############################################################
##Full Power Spectral Density Plots 
##Compares cases of misclassifed signals (see Fig 5 in paper)
#############################################################

wav_list = ['stored_sounds/PIPI_20160607_223337-002_1304_5_1_6_B_.wav',
    'stored_sounds/PIPY_20160607_221032-007_138_5_1_2_B_.wav']

threshold = 0.1
min_f = 45
max_f = 60
binwidth = 0.9

##Typical examples of misclassifiations for each case
wav_input = ['typical_misclass/PIPY_20160820_222844-010_681_5_1_6_B_.wav',
    'typical_misclass/PIPI_20160607_230607-009_485_5_1_4_B_.wav',
    'typical_misclass/PIPI_20160825_214054-036_984_5_1_8_B_.wav',
    'typical_misclass/PIPY_20160919_224551-012_1568_1_1_8_B_.wav']

bins=np.arange((min_f),(max_f+1), step=binwidth)
bins=np.around(bins, decimals=2, out=None)

fig, ax = plt.subplots(2,2, figsize=(14,9))  

tf2, yf2, tf3, yf3, peaks, maxval = power_spec(wav_list[0], threshold)
for i in range(2):
    for j in range(2):
        ax[i,j].plot(tf2,yf2/maxval, label='Stored PIPI', color='red', zorder=2)
        

tf2, yf2, tf3, yf3, peaks, maxval = power_spec(wav_list[1], threshold)
for i in range(2):
    for j in range(2):
        ax[i,j].plot(tf2,yf2/maxval, label='Stored PIPY', color='green',zorder=2)
        

count = 0
labs = ['PIPY detected as PIPI','PIPI detected as PIPY',
    'PIPI detected as UnID','PIPY detected as UnID']
for i in range(2):
    for j in range(2):
        if count == 0:
            PIPlist = glob.glob('PIPI/PIPY*')
            print(count, len(PIPlist))
        elif count == 1:
            PIPlist = glob.glob('PIPY/PIPI*')
            print(count, len(PIPlist))
        elif count == 2:
            PIPlist = glob.glob('UNID/PIPI*')
            print(count, len(PIPlist))
        elif count == 3:
            PIPlist = glob.glob('UNID/PIPY*')
            print(count, len(PIPlist))
        
        for wav in PIPlist:
            tf2, yf2, tf3, yf3, peaks, maxval = power_spec(wav, threshold)
            if maxval != 0:
                ax[i,j].plot(tf2,yf2/maxval, color='lightblue', zorder=0, lw='0.3')
            

        ax[i,j].plot(np.NaN, np.NaN, '-', color='lightblue', label=labs[count])
        tf2, yf2, tf3, yf3, peaks, maxval = power_spec(wav_input[count], threshold)
        ax[i,j].plot(tf2,yf2/maxval, color='blue', zorder=1, label='Typical example')     
        ax[i,j].legend(loc='upper right')
        ax[i,j].set_xlim(min_f,max_f)
        ax[i,j].set_ylim(-0.03,1.02)
        ax[i,j].set_xticks(bins)
        ax[i,j].set_xticklabels(bins, rotation=45)
        ax[i,j].grid(axis = 'x', color = '0.80')
        ax[i,j].set_xlabel('Frequency (kHz)')
        ax[i,j].set_ylabel('Relative Power')

        count += 1

fig.tight_layout(pad=0.1)
fig.savefig('power_spec_plot_misclass.png', dpi=600)

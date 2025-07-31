##################################################
#BRaiN_bioacoustics EI 2025. Analyse/Plot Results.
##################################################


import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import glob



def sounds_processor2(wav_list, threshold, min_f, max_f):
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

        #find the neurons which are on:
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



def power_spec(wav, threshold):
    global lf, hf, binwidth

    s_rate, signal = wavfile.read(wav)

    # Number of sample points
    N = len(signal)
    # Sample period
    T = 1.0 / s_rate
      
    yf = fft(signal)
    tf = fftfreq(N, T)

    yf2 = np.abs(yf[0:N//2])**2
    tf2 = tf[0:N//2]*1.e-3
        
    peaks_inrange = np.where((tf2 > lf) & (tf2 < hf))

    yf3 = yf2[peaks_inrange]
    tf3 = tf2[peaks_inrange]
    peaks, _ = find_peaks(yf3, height=0)
        
    maxval = max(yf3[peaks], default = 0) 
        
    TH = threshold*maxval
        
    peaks, _ = find_peaks(yf3, height=TH)

    return tf2, yf2, tf3, yf3, peaks, maxval


#########################################################################

wav_list = ['stored_sounds/PIPI_20160607_223337-002_1304_5_1_6_B_.wav',
    'stored_sounds/PIPY_20160607_221032-007_138_5_1_2_B_.wav']

threshold = 0.1
min_f = 45
max_f = 60
peaks_f = sounds_processor2(wav_list, threshold, min_f, max_f)

neurons, binwidth, lf, hf = network_model(peaks_f)


wav_input = ['typical_misclass/PIPY_20160820_222844-010_681_5_1_6_B_.wav',
    'typical_misclass/PIPI_20160607_230607-009_485_5_1_4_B_.wav',
    'typical_misclass/PIPI_20160825_214054-036_984_5_1_8_B_.wav',
    'typical_misclass/PIPY_20160919_224551-012_1568_1_1_8_B_.wav']



print(wav_input)

bins=np.arange((lf),(hf+1), step=binwidth)
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
        ax[i,j].set_xlim(lf,hf)
        ax[i,j].set_ylim(-0.03,1.02)
        ax[i,j].set_xticks(bins)
        ax[i,j].set_xticklabels(bins, rotation=45)
        ax[i,j].grid(axis = 'x', color = '0.80')
        ax[i,j].set_xlabel('Frequency (kHz)')
        ax[i,j].set_ylabel('Relative Power')

        count += 1

fig.tight_layout(pad=0.1)
fig.savefig('power_spec_plot_misclass.png', dpi=600)
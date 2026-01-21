

#######################
### BRaiNN LEMUR 1  ###
#######################

##Importing the necessary libraries and packages in python.
import glob
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import awkward as ak
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


def stored_processor(wav_list):
    global min_f, max_f, threshold
    
    peaks_fv =[]
    min_val = []
    max_val = []
    count = 0
    for wav in wav_list:
        
        s_rate, signal = wavfile.read(wav)
        
        if len(signal.shape)==2:
            signal = signal[:,0]
        if len(signal.shape)==1:
            signal = signal

        N = len(signal)  # Number of sample points
        T = 1.0 / s_rate # Sample period 
        
        yf = fft(signal)
        tf = fftfreq(N, T)

        yf2 = np.abs(yf[0:N//2])**2 # power
        tf2 = tf[0:N//2]*1.e-3   # kHz
        
        peaks_inrange = np.where((tf2 > min_f) & (tf2 < max_f))

        yf3 = yf2[peaks_inrange]
        
        tf3 = tf2[peaks_inrange]
        peaks, _ = find_peaks(yf3, height=0)
               
        maxval = max(yf3[peaks], default=0)
        max_val.append(maxval)
        
        TH = threshold*maxval
        
        minval = min(yf3[peaks])
        min_val.append(minval)
        
        peaks, _ = find_peaks(yf3, height=TH)
        
        peaks_f_values = tf3[peaks]
                   
        peaks_f_values = np.delete(peaks_f_values, np.where(peaks_f_values < min_f))
        peaks_f_values = np.delete(peaks_f_values, np.where(peaks_f_values > max_f))
        
        peaks_values=ak.Array(peaks_f_values)
        peaks_fv=ak.concatenate([peaks_fv, peaks_values[np.newaxis]], axis=0)
               
        count += 1
        

    counts = ak.max(ak.count(peaks_fv, axis=-1))
    counts = int(counts)
    peaks_fv1=np.asmatrix(ak.fill_none(ak.pad_none(peaks_fv, counts, clip=True), 999))
    v = np.asmatrix(peaks_fv[:,-1])
    v=v.T
    peaks_fv1 = np.where(peaks_fv1==999, v, peaks_fv1)
    
    return min_val, max_val, peaks_fv1


def sounds_processor(wav_list, sl_width):
    global min_f, max_f, threshold, minval
    peaks_fv =[]
    count = 0
    
    df = pd.DataFrame(columns=['filename', 'total_windows', 'file_window_no'])
    
    for wav in wav_list:
        
        #file name with extension
        filename = os.path.basename(wav)
        # file name without extension
        filename = os.path.splitext(filename)[0]
               
        s_rate, signal = wavfile.read(wav)
        
        if len(signal.shape)==2:
            signal = signal[:,0]
        if len(signal.shape)==1:
            signal = signal

        N = len(signal)  # Number of sample points
        T = 1.0 / s_rate # Sample period
        
        W = sl_width*s_rate #
        win = round(N/W)
        
        count1 = 0
        for i in range(win):
            window = signal[int(i*W):int((i+1)*W)]
            
            w = len(window)
            yf = fft(window)
            tf = fftfreq(w, T) 

            yf2 = np.abs(yf[0:w//2])**2 # power
            tf2 = tf[0:w//2]*1.e-3   # kHz
            
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
        
            peaks_values=ak.Array(peaks_f_values)
            peaks_fv=ak.concatenate([peaks_fv, peaks_values[np.newaxis]], axis=0)
            
            #show window in dataframe as time from start of file
            
            sl = count1
            sl = sl*sl_width
            sl = format(sl, '.2f')
            
            df2 = pd.DataFrame([[filename, win, sl]], columns = ['filename', 'total_windows','file_window_no'])
            df = df._append(df2)
            
            
            count1 += 1
            count += 1
        
        
    counts = ak.max(ak.count(peaks_fv, axis=-1))
    counts = int(counts)
    peaks_fv1=np.asmatrix(ak.fill_none(ak.pad_none(peaks_fv, counts, clip=True), 999))
    v = np.asmatrix(peaks_fv[:,-1])
    v=v.T
    peaks_fv1 = np.where(peaks_fv1==999, v, peaks_fv1)  
    
    return count, df, peaks_fv1



def neuron_firing(n_signals, peaks_f, freqrange, bins):
    
    neurons = np.zeros((n_signals, len(freqrange)))-1
       
    for j in range(n_signals):
        
        peaks_f1 = peaks_f[j]
        
        if np.all(peaks_f1==0):
            neurons[j,:] =0
            continue
        
        ##find the neurons which are on:
        index = np.digitize(peaks_f1, bins)-1
        neurons[j,index]=1
               
    return neurons



def network_model(peaks_f):
    global min_f, max_f, binwidth
    
    freqrange = np.arange(min_f,max_f+binwidth,binwidth)
    bins = np.arange(freqrange[0]-binwidth/2,freqrange[-1]+binwidth/2,binwidth)
       
    dimensions = peaks_f.shape
    rows, columns = dimensions
    maxID = rows-1
    
    neurons = neuron_firing(maxID+1, peaks_f, freqrange, bins)
    
    return neurons, freqrange, bins




##########################
# Hopfield Network Model #
##########################

## Global parameters
threshold = 0.1
min_f = 0
max_f = 1.3
binwidth = 0.1
sl_width = 1 # time slice
iterations = 4

###########################################
# Storing signals into the network memory #
###########################################

##Assigning the number of  neurons as well as the 
##rows and columns for the matrix.

wav_list = glob.glob("lemur_calls/*.wav")

min_val, max_val, peaks_fv = stored_processor(wav_list)

minval = min(min_val)
max_val = min(max_val)

neurons, freqrange, bins = network_model(peaks_fv)

##Index neurons by call type file label
neurons = pd.DataFrame(neurons)
labels = ['']*len(wav_list)

for i,wav in zip(range(len(wav_list)),wav_list):
    labels[i] = wav
neurons.index = labels


sf = neurons.iloc[:,:].values

N = len(sf[0])
sf = np.asmatrix(sf)
row_count = len(sf)

W = np.zeros((N,N))
   
for i in range(row_count):    
    p = sf[i]
    p = p.reshape(N,1)
    W += p*p.T-np.identity(N)


####################################################################   
# Classifying Signals; .wav files are provided by the user.        #
#                                                                  #
# You will need to ensure you have created a folder called 'data'  #
# which contains the .wav files you wish the model to classify.    #
####################################################################


wav_input = glob.glob("data/*.wav")
n_signals, df, peaks_fv = sounds_processor(wav_input, sl_width)
input_signal = neuron_firing(n_signals, peaks_fv, freqrange, bins)

x = np.asmatrix(input_signal)
x = x.T

z = np.zeros(N)

for k in range(iterations):
    z = np.dot(W, x)
    x = np.sign(z)


##Sort files by call type. N.B. in this early iteration of the code you will have
##to manually check the lines of code below, if you are changing the stored sounds.
grumble = neurons.iloc[0,:].values
print('GRUMBLE', grumble)
alarm = neurons.iloc[1,:].values
print('ALARM', alarm)


df3 = pd.DataFrame(columns=['call'])

for i in range(n_signals):
    
    if (x[:,i].T == grumble).all():
        df4 = pd.DataFrame([['L_grumble']], columns=['call'])
        df3 = df3._append(df4)
    
    elif (x[:,i].T == alarm).all():
        df4 = pd.DataFrame([['L_alarm']], columns=['call'])
        df3 = df3._append(df4)
    
    else:
        df4 = pd.DataFrame([['UNID']], columns=['call'])
        df3 = df3._append(df4)


df = pd.concat([df, df3], axis=1)
df = df.reset_index(drop=True)
df["file_window_no"] = pd.to_numeric(df["file_window_no"])
df.to_csv('BRaiNN_class_centric.csv', index=True)

df4 = df.pivot(index=('filename', 'total_windows'), columns='file_window_no', values='call') 
df4.to_csv('BRaiNN_file_centric.csv', index=True)



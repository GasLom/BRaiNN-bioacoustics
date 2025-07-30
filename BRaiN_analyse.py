
###############################################
## BRaiN_bioacoustics EI 2025. Analyse Results.
###############################################

##Importing the necessary libraries and packages in python.
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
#from os import sys
import glob
import re

# Plot stored sounds and sound to be examined

def main():
    input("Press enter and exit")

def plot(wav, wav_list_s, filename):
    plt.figure()
    wav_plot_list = []    
    wav_plot_list.extend(wav_list_s)
    wav_plot_list.extend([wav])
    for wav in wav_plot_list:
        s_rate, signal = wavfile.read(wav)

        # Number of sample points
        N = len(signal)
        # Sample period
        T = 1.0 / s_rate
  
        yf = fft(signal)
        tf = fftfreq(N, T)

        yf2 = np.abs(yf[0:N//2])**2
        tf2 = tf[0:N//2]*1.e-3

        maxval = max(yf2)
        TH = 0.1*maxval
    
        peaks, _ = find_peaks(yf2, height=TH)
        peaks_f_values = tf2[peaks]

        peak_f = peaks_f_values

        plt.subplot()
        plt.plot(tf2,yf2/maxval)
        plt.plot(tf2[peaks], yf2[peaks]/maxval, 'x', color = 'black')
        plt.xlim((lf),(hf))
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Relative Power')
        #filename = 'Stored Sounds and ', wav[-41:-4]
        
        plt.grid(axis = 'x', color = '0.80 ')
        plt.xticks(np.arange((lf),(hf), step=0.9), rotation=45)   
        plt.title(filename)
        
    plt.show()
        #plt.savefig(filename+'.png')
    return

# Find freq of highest peak

def maxF (wav):
    
    s_rate, signal = wavfile.read(wav)
    # Number of sample points
    N = len(signal)
    # Sample period
    T = 1.0 / s_rate
    yf = fft(signal)
    tf = fftfreq(N, T)
    
    yf2 = np.abs(yf[0:N//2])**2
    tf2 = tf[0:N//2]*1.e-3
    maxval = max(yf2)
    TH = 0.1*maxval
    peaks, _ = find_peaks(yf2, height=TH)
    
    max_peak = peaks[np.argmax(yf2[peaks])]
    max_peak_f = tf2[max_peak]

    return max_peak_f


# Analyse signals by max_peak_f

def analyse(max_peak_f, count1, count2, count3, count4):
    
    if 43 < max_peak_f < 49.5: #< 49.8:
        #input(f"Peak frequency is {max_peak_f}. Press enter to continue:")
        count1 +=1
        
    if 50.5 < max_peak_f < 60: #< 49.8:
        #input(f"Peak frequency is {max_peak_f}. Press enter to continue:")
        count2 +=1
        
    if 49.5 <= max_peak_f <= 50.5: #< 49.8:
        #input(f"Peak frequency is {max_peak_f}. Press enter to continue:")
        count3 +=1
        
    elif 43 >= max_peak_f or max_peak_f >= 60:
        count4 +=1
        
    return count1, count2, count3, count4
    

grid = 16 #grid lines to make plots easier to analyse
lf = 45
hf = 60
freqrange = np.linspace(lf,hf,grid)


### N.B. We recommend printing results for each folder in separate runs, therefore 
### comment out plots which are not required or use input("Press enter to continue:")



##########################
# Pi in Py
##########################

wav_list = glob.glob("PIPY/PIPI*.wav")

file_no = len(wav_list)

wav_list_s = ['stored_sounds/PIPI_20160607_223337-002_1304_5_1_6_B_.wav',
            'stored_sounds/PIPY_20160607_221032-007_138_5_1_2_B_.wav']


pi_in_py_actual_pi = 0
pi_in_py_actual_py = 0
pi_in_py_unclass = 0
other = 0

for wav in wav_list:
    
    max_peak_f = maxF(wav)
    
    pi_in_py_actual_pi, pi_in_py_actual_py, pi_in_py_unclass, other = analyse(max_peak_f, pi_in_py_actual_pi, pi_in_py_actual_py, 
    pi_in_py_unclass, other)
    
    filename = 'Pi in Py: ', wav[-42:-4]
    plots = plot(wav, wav_list_s, filename)  

#sys.exit()
print('\nNumber of PIPI classified PIPY: ', file_no, '\n')       
print(f'        Actual Pi in Py - {pi_in_py_actual_pi}.')
print(f'   Mislabelled Pi in Py - {pi_in_py_actual_py}.')
print(f'Unclassifiable Pi in Py - {pi_in_py_unclass}.')     
print(f'            Other in Py - {other}.')       

#input("Press enter to continue:")


##########################
# Py in Pi
##########################

wav_list = glob.glob("PIPI/PIPY*.wav")

file_no = len(wav_list)

wav_list_s = ['stored_sounds/PIPI_20160607_223337-002_1304_5_1_6_B_.wav',
            'stored_sounds/PIPY_20160607_221032-007_138_5_1_2_B_.wav']


py_in_pi_actual_pi = 0
py_in_pi_actual_py = 0
py_in_pi_unclass = 0
other_in_pi = 0

for wav in wav_list:
    
    max_peak_f = maxF(wav)
    
    py_in_pi_actual_pi, py_in_pi_actual_py, py_in_pi_unclass, other = analyse(max_peak_f, 
    py_in_pi_actual_pi, py_in_pi_actual_py, py_in_pi_unclass, other)

    filename = 'Py in Pi: ', wav[-43:-4]
    #plots = plot(wav, wav_list_s, filename)  

#sys.exit()
print('\n\nNumber of PIPY classified PIPI: ', file_no, '\n')       
print(f'   Mislabelled Py in Pi - {py_in_pi_actual_pi}.')
print(f'        Actual Py in Pi - {py_in_pi_actual_py}.')
print(f'Unclassifiable Py in Pi - {py_in_pi_unclass}.')     
print(f'            Other in Pi - {other_in_pi}.')       

#input("Press enter to continue:")


##########################
# Pi and Py in Un
##########################


wav_list = glob.glob("UNID/*.wav")

file_no = len(wav_list)

wav_list_s = ['stored_sounds/PIPI_20160607_223337-002_1304_5_1_6_B_.wav',
            'stored_sounds/PIPY_20160607_221032-007_138_5_1_2_B_.wav']

count_py_in_un_actual_pi = 0
count_py_in_un_actual_py = 0
count_pi_in_un_actual_pi = 0
count_pi_in_un_actual_py = 0
count_pi_or_py_unclassifiable = 0
not_pip_in_un = 0

for wav in wav_list:
    
    max_peak_f = maxF(wav)         
        
    if 43 < max_peak_f < 49.5: #< 49.8:
        match = re.search("/PIPI_", wav)
        if match:
            count_pi_in_un_actual_pi +=1 
        match = re.search("/PIPY_", wav)
        if match:
            count_py_in_un_actual_pi +=1
            
    if 50.5 < max_peak_f < 60: #< 49.8:
        match = re.search("/PIPY_", wav)
        if match:
            count_py_in_un_actual_py +=1
        match = re.search("/PIPI_", wav)
        if match:
            count_pi_in_un_actual_py +=1
        #input(f"Peak frequency is {max_peak_f}. Press enter to continue:")
        #print(wav)
    
    if 43 >= max_peak_f or max_peak_f >= 60:
        not_pip_in_un +=1
        
    elif 49.5 <= max_peak_f <= 50.5: #< 49.8:
        count_pi_or_py_unclassifiable +=1
    
    filename = 'Pi/Py in UnID: ', wav[-42:-4]
    #plots = plot(wav, wav_list_s, filename)   
          
#sys.exit()
print('\n\nNumber of PIPI and PIPY in UNIDENTIFIED: ', file_no, '\n')    
print(f'              Actual Py in Unid - {count_py_in_un_actual_py }.')
print(f'Mislabelled Py (act Pi) in Unid - {count_py_in_un_actual_pi}.')
print(f'              Actual Pi in Unid - {count_pi_in_un_actual_pi}.')
print(f'Mislabelled Pi (act Py) in Unid - {count_pi_in_un_actual_py}')
print(f'Unclassifiable Py or Pi in Unid - {count_pi_or_py_unclassifiable}.')     
print(f'                Not PIP in Unid - {not_pip_in_un}.') 
      


### ALSO ANALYSE PI AND PY IN CORRECT FOLDERS?

##########################
# PY in Py
##########################

wav_list = glob.glob("PIPY/PIPY*.wav")

file_no = len(wav_list)

wav_list_s = ['stored_sounds/PIPI_20160607_223337-002_1304_5_1_6_B_.wav',
            'stored_sounds/PIPY_20160607_221032-007_138_5_1_2_B_.wav']

py_in_py_actual_py = 0
py_in_py_actual_pi = 0
py_in_py_unclass = 0
other = 0

for wav in wav_list:
    
    max_peak_f = maxF(wav)
    
    py_in_py_actual_pi, py_in_py_actual_py, py_in_py_unclass, other = analyse(max_peak_f, py_in_py_actual_pi, py_in_py_actual_py, 
    py_in_py_unclass, other)
    filename = 'Py in Py: ', wav[-42:-4]
    #plots = plot(wav, wav_list_s, filename)

#sys.exit()
print('\nNumber of PIPY classified PIPY: ', file_no, '\n')       
print(f'        Actual Py in Py - {py_in_py_actual_py}.')
print(f'      Pi mislabelled Py - {py_in_py_actual_pi}.')
print(f'Unclassifiable Py in Py - {py_in_py_unclass}.')     
print(f'            Other in Py - {other}.')       

#input("Press enter to continue:")




##########################
# PI in PI
##########################

wav_list = glob.glob("PIPI/PIPI*.wav")

file_no = len(wav_list)

wav_list_s = ['stored_sounds/PIPI_20160607_223337-002_1304_5_1_6_B_.wav',
            'stored_sounds/PIPY_20160607_221032-007_138_5_1_2_B_.wav']

pi_in_pi_actual_pi = 0
pi_in_pi_actual_py = 0
pi_in_pi_unclass = 0
other = 0

for wav in wav_list:
    
    max_peak_f = maxF(wav)
    
    pi_in_pi_actual_pi, pi_in_pi_actual_py, pi_in_pi_unclass, other = analyse(max_peak_f, pi_in_pi_actual_pi, pi_in_pi_actual_py, 
    pi_in_pi_unclass, other)
    filename = 'PI in PI: ', wav[-43:-4]
    #plots = plot(wav, wav_list_s, filename)
    #if 43 >= max_peak_f or max_peak_f >= 60:
    #    print(max_peak_f)
    #    input(f"Max Peak F in {wav } is {max_peak_f}. Press enter to continue:")

#sys.exit()
print('\nNumber of PIPI classified PIPI: ', file_no, '\n')       
print(f'        Actual Pi in Pi - {pi_in_pi_actual_pi}.')
print(f'      Py mislabelled Pi - {pi_in_pi_actual_py}.')
print(f'Unclassifiable Pi in Pi - {pi_in_pi_unclass}.')     
print(f'            Other in Pi - {other}.')       

#input("Press enter to continue:")


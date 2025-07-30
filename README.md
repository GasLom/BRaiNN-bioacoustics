# BRaiN_bioacoustics
First-of-its-kind AI model for bioacoustic detection using a lightweight associative memory Hopfield neural network (bioacoustic recognition AI network).

## Requirements:

matplotlib==3.10.3  
numpy==2.3.2  
pandas==2.3.1  
psutil==5.9.0  
scikit_learn==1.7.1  
scipy==1.16.1  


## Instructions:

The data used to introduce this model in Ecological Informatics (citation pending) is available at https://zenodo.org/records/3247097. To replicate the results of the EI paper, download the dataset and follow the instructions within the paper and the labelling on Zenodo to remove the short silence files. 

For the simplest execution put all 10,384 wav files in a folder named ‘DATA’ in the same location as the contents of the code zip file. Remember to remove the two files currently in stored_sounds or you will end up with 10,386 wav files. Set up empty folders in the same location: PIPI’, ‘PIPY’, ‘SILENCE’, ‘UNID’, ‘UNCLASS’. To classify the data run BRaiN.py. Run the BRaiN_analyse.py once the main classification code (BRaiN.py) has run. Before a re-run return files to the DATA folder using BRaiN_fileshifter.py. N.B. Faster runtimes can be achieved in a terminal and without shifting files, however the runtimes reported in the paper include the time taken to move files before analysis.

<div align="center">
  <h1>BRaiN_bioacoustics</h1>
       <img src="https://github.com/GasLom/BRaiN_bioacoustics/blob/main/BRaiN.png?raw=true" width="300" alt=“BRaiN-Logo" />
    </a>
</div>
<br>
<div align="center">

</div>

First-of-its-kind AI model for bioacoustic detection using a lightweight associative memory Hopfield neural network (bioacoustic recognition AI network).

## Requirements:

matplotlib==3.10.3  
numpy==2.3.2  
pandas==2.3.1  
psutil==5.9.0  
scikit_learn==1.7.1  
scipy==1.16.1  


## Instructions:

The dataset used to develop the model introduced in Ecological Informatics (citation pending) is available at https://zenodo.org/records/3247097. To replicate the results of the EI paper, download the dataset and follow the instructions within the paper and the labelling on Zenodo to remove the short silence files, i.e. remove all wav files from the downloaded SILENCES folder that DO NOT start with S-.

For the simplest execution put the remaining 10,384 wav files in a folder named ‘DATA’ in the same location as the contents of the code zip file. Remember to remove the two files currently in stored_sounds or you will end up with 10,386 wav files. Set up empty folders in the same location: PIPI’, ‘PIPY’, ‘SILENCE’, ‘UNID’, ‘UNCLASS’. To classify the data run BRaiN.py. Run the BRaiN_analyse_plot.py once the main classification code (BRaiN.py) has run. Before a re-run, return files to the DATA folder using BRaiN_fileshifter.py. Double check your folders are empty and DATA contains the full dataset before running again. N.B. Faster runtimes can be achieved in a terminal and without shifting files, however the runtimes reported in the paper include the time taken to move files before analysis.

To analyse/classify longer files or for different species do get in touch with the developers and they will be happy to advise on later iterations of the code. They are currently working with captive and wild primate datasets.  

## Acknowledgements:

The authors would like to thank the [OpenBright Foundation](https://openbright.org.uk/) and trustee Elizabeth Molyneux for their support and funding. We also thank the [University of Wolverhampton](https://www.wlv.ac.uk/) for the Invest to Grow PhD Studentship funding. Finally, thanks also go to Catherine Povey ([Just Mammals Ltd](https://www.justmammals.co.uk/)) for her expertise and useful discussions.  


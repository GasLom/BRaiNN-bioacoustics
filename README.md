<div align="center">
  <h1>BRaiNN_bioacoustics</h1>
       <img src="https://github.com/GasLom/BRaiN_bioacoustics/blob/main/BRaiN.png?raw=true" width="300" alt=“BRaiN-Logo" />
    </a>
</div>
<br>


The **B**ioacoustic **R**ecognition AI **N**eural **N**etwork (BRaiNN) team is made up of a group of researchers at the [University of Wolverhampton](https://www.wlv.ac.uk/), lead by [Dr Andrew Gascoyne](https://researchers.wlv.ac.uk/A.D.Gascoyne) and [Wendy Lomas](https://researchers.wlv.ac.uk/W.Lomas), developing AI models for bioacoustic analysis. This repository provides the software to detect common and soprano pipistrelles using a lightweight associative memory Hopfield neural network. 

Feel free to use the software for your acoustic analyses and research. If you do, please cite as:

```bibtex
@article{GasLom2025BRaiNN,
  title={First-of-its-kind AI model for bioacoustic detection using a lightweight associative memory Hopfield neural network},
  author={Gascoyne, Andrew and Lomas, Wendy},
  journal={Ecological Informatics},
  volume={},
  pages={},
  year={2025},
  publisher={Elsevier}
}
```

## Requirements:

matplotlib==3.10.3  
numpy==2.3.2  
pandas==2.3.1  
psutil==5.9.0  
scikit_learn==1.7.1  
scipy==1.16.1  


## Instructions:

The dataset used to develop the model is available on [Zenodo](https://zenodo.org/records/3247097). To replicate the results of this paper (citation pending), download the Zenodo dataset, paying careful attention to the filename syntax, and follow the instructions within the paper. It is important to remove all wav files from the downloaded SILENCES folder with filenames that DO NOT begin with S-, otherwise this will over inflate the model performance metrics, see paper for details.

For the simplest execution:

* Place the remaining 10,384 wav files in a folder named 'DATA’ in the same location as the contents of the code zip file. Remember to remove the two files currently in stored_sounds or you will end up with 10,386 wav files.
* Set up five empty folders in the same location: 'PIPI’, 'PIPY’, 'SILENCE’, 'UNID’, and 'UNCLASS’.
* To classify the raw data run BRaiNN.py
* To examine the results run BRaiNN_analyse_plot.py once the main classification code (BRaiNN.py) has run.
* N.B. The fastest runtimes can be achieved in a command-line interface and without shifting files, however the runtimes reported in the paper include the time taken to move files before analysis.
<!-- 
* Before a re-run, return files to the DATA folder using BRaiNN_fileshifter.py. Double check your folders are empty and DATA contains the full dataset (10,384 wav files) before running again.
-->

To analyse/classify longer files or for different species do get in touch with the developers and they will be happy to advise on later iterations of the code. They are currently working with captive and wild primate datasets.  

## Acknowledgements:

The authors would like to thank the [OpenBright Foundation](https://openbright.org.uk/) and trustee Elizabeth Molyneux for their support and funding, and the [University of Wolverhampton](https://www.wlv.ac.uk/) for the Invest to Grow PhD Studentship funding. Finally, thanks also go to Catherine Povey ([Just Mammals Ltd](https://www.justmammals.co.uk/)) for her expertise and useful discussions.  


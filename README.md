<div align="center">
  <h1>BRaiNN-bioacoustics</h1>
       <img src="https://github.com/GasLom/BRaiN_bioacoustics/blob/main/BRaiNN_sticker.png?raw=true" width="300" alt=“BRaiNN-Logo" />
    </a>
</div>
<br>


The **B**ioacoustic **R**ecognition AI **N**eural **N**etwork (BRaiNN) team is made up of a group of researchers at the [University of Wolverhampton](https://www.wlv.ac.uk/), led by [Andrew Gascoyne](https://researchers.wlv.ac.uk/A.D.Gascoyne) and [Wendy Lomas](https://researchers.wlv.ac.uk/W.Lomas), developing lightweight AI models for bioacoustic analysis. This repository provides the software to detect common and soprano pipistrelles using a lightweight associative memory Hopfield neural network. 

Feel free to use the software for your acoustic analyses and research. If you do, please cite as:

```bibtex
@article{GasLom2025BRaiNN,
  title={First-of-its-kind AI model for bioacoustic detection using a lightweight associative memory Hopfield neural network},
  author={Gascoyne, Andrew and Lomas, Wendy},
  journal={Ecological Informatics},
  volume={91},
  pages={103382},
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

To replicate the results of this paper (Gascoyne and Lomas, 2025) follow these instructions for the simplest execution:

* Create an empty folder, 'DATA' and then unzip all the dataset files (Gascoyne and Lomas, 2025; Bertran et al, 2019). Move all the wav files from PIPI_paper, PIPY_paper, SILENCE_paper and SILENCE2_paper to the DATA folder.
* Set up five empty classification folders in the same location: 'PIPI’, 'PIPY’, 'SILENCE’, 'UNID’, and 'UNCLASS’.
  * PIPI: common pipistrelle
  * PIPY: soprano pipistrelle
  * SILENCE: pre-processed silences
  * UNID: unidentified, i.e. model converges to a spurious state
  * UNCLASS: pre-processed unclassifiable files of echolocation pulses with FmaxE between 49 and 51 kHz (Russ, 2021; Aughney et al., 2018; Catto et al., 2003)
* To classify the raw data run BRaiNN.py
* To examine the results run BRaiNN_analyse_plot.py once the main classification code (BRaiNN.py) has run.
* N.B. The fastest runtimes can be achieved in a command-line interface and without shifting files, however the runtimes reported in the paper include the time taken to move files before analysis.

To analyse/classify longer files or for different species do get in touch with the team and they will be happy to advise on later iterations of the software. We are currently working with captive and wild primate datasets.  


## References:

Aughney, T., Roche, N., Langton, S., 2018. The Irish Bat Monitoring Programme 2015-2017. Irish Wildlife Manuals, No. 103, National Parks and Wildlife Service. Department of Culture, Heritage and the Gaeltacht, Ireland.

Bertran, M., Alsina-Pages, R., Tena, E., 2019. Pipistrellus pipistrellus and pipistrellus pygmaeus in the iberian peninsula: An annotated segmented dataset and a proof of concept of a classifier in a real environment. Applied Sciences 9. doi:10.3390/app9173467.

Catto, C., Coyte, A., Agate, J., Langton, S., 2003. Bats as Indicators of Environmental Quality R&D Technical Report E1-129/TR. Technical Report ISBN: 1 844 32251 3. Environment Agency. Rio House, Waterside Drive, Aztec West, Almondsbury Bristol BS32 4UD. N.B. This document was produced under R&D Project E1-129 by the Bat Conservation Trust.

Gascoyne, A., Lomas, W., 2025. First-of-its-kind AI model for bioacoustic detection using a lightweight associative memory Hopfield neural network. Ecological Informatics, vol. 91, pp 103382, (2025). doi.org/10.1016/j.ecoinf.2025.103382.

Russ, J., 2021. Bat calls of Britain and Europe. Bat Biology and Conservation, Pelagic Publishing, Exeter, England.


## Acknowledgements:

The authors would like to thank the [OpenBright Foundation](https://openbright.org.uk/) and trustee Elizabeth Molyneux for their support and funding, and the [University of Wolverhampton](https://www.wlv.ac.uk/) for the Invest to Grow PhD Studentship funding. Finally, thanks also go to Catherine Povey ([Just Mammals Ltd](https://www.justmammals.co.uk/)) for her expertise and useful discussions.  


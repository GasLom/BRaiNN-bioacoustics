<div align="center">
  <h1>BRaiNN-bioacoustics</h1>
       <img src="https://github.com/GasLom/BRaiN_bioacoustics/blob/main/BRaiNN_sticker.png?raw=true" width="300" alt=“BRaiNN-Logo" />
    </a>
</div>
<br>


The **B**ioacoustic **R**ecognition AI **N**eural **N**etwork (BRaiNN) team is made up of a group of researchers at the [University of Wolverhampton](https://www.wlv.ac.uk/) led by [Andrew Gascoyne](https://researchers.wlv.ac.uk/A.D.Gascoyne) and [Wendy Lomas](https://researchers.wlv.ac.uk/W.Lomas). The team is developing lightweight AI models for bioacoustic analysis. This repository provides the software to detect a variety of species specific call types and will be updated as the research progresses. Currently we have models available here to predict common and soprano pipistrelles as well as black and white ruffed lemurs using lightweight associative memory Hopfield neural networks. Feel free to use the software for your acoustic analyses and research. If you do, please use the appropriate citations below.

For the original model, which was used to detect two bat species, use:
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

For the model used to detect lemur call types, please use:
```bibtex
@InProceedings{LomGDVN2025BRaiNN,
  title={Lightweight Hopfield Neural Networks for Bioacoustic Detection and Call Monitoring of Captive Primates},
  author={Lomas, Wendy, Gascoyne, Andrew, Dubreuil, Colin, Vaglio, Stefano and Naughton, Liam}
  editor={Arai, Kohei},
  booktitle={Proceedings of the Future Technologies Conference (FTC) 2025, Volume 1},
  pages={603--617},
  year={2025},
  publisher={Springer Nature Switzerland} 
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

To replicate the results in Gascoyne and Lomas (2025) and Lomas et al. (2025) follow the instructions below.

### Gascoyne and Lomas (2025)

For the simplest execution:

* Create an empty folder, 'DATA' and then unzip all the dataset files (Gascoyne and Lomas, 2025; Bertran et al, 2019). Move all the wav files from 'PIPI_paper', 'PIPY_paper', 'SILENCE_paper' and 'SILENCE2_paper' to the 'DATA' folder.
* Set up five empty classification folders in the same location: 'PIPI’, 'PIPY’, 'SILENCE’, 'UNID’, and 'UNCLASS’.
  * PIPI: common pipistrelle
  * PIPY: soprano pipistrelle
  * SILENCE: pre-processed silences
  * UNID: unidentified, i.e. model converges to a spurious state
  * UNCLASS: pre-processed unclassifiable files of echolocation pulses with FmaxE between 49 and 51 kHz (Russ, 2021; Aughney et al., 2018; Catto et al., 2003)
* To classify the raw data run BRaiNN.py
* To examine the results run BRaiNN_analyse_plot.py once the main classification code (BRaiNN.py) has run.
* N.B. The fastest runtimes can be achieved in a command-line interface and without shifting files, however the runtimes reported in the paper include the time taken to move files before analysis.


### Lomas et al. (2025)

In this paper two models were developed. Model 1 was trained on two representative black and white ruffed lemur calls characterised as an alarm call and a grumble (located in folder 'lemur_calls'). Model 2 was also trained on these calls and an additional signal representative of noise made by the lemurs as the moved around their apparatus near the recorder, see 'lemur_calls_noise'. To train and classify using model 1 run BRaiNN_lemur_1.py, and to train and classify using model 2 run BRaiNN_lemur_2.py. 

N.B. You will have to provide signals to be classified via the code instructions. We would also recommend using your own noise signal for model 2, representative of sounds you wish to filter out. Please see the paper for more details. The full dataset used in this research for evaluation is currently unavailable here as we were not the only curators, but we hope to update this in the future.

&nbsp;
&nbsp;

## The Team at the University of Wolverhampton:

To analyse/classify files from different species or with different parameters do get in touch with the team and they will be happy to advise on later iterations of the software. We are currently working with captive and wild primate and bat datasets.  

Andrew Gascoyne, School of Engineering, Computing and Mathematical Sciences (a.d.gascoyne@wlv.ac.uk).

Wendy Lomas, School of Engineering, Computing and Mathematical Sciences (w.k.lomas@wlv.ac.uk)

Colin Dubreuil, School of Pharmacy and Life Sciences (c.dubreuil@wlv.ac.uk)

Stefano Vaglio, School of Pharmacy and Life Sciences (s.vaglio@wlv.ac.uk)

Liam Naughton, School of Engineering, Computing and Mathematical Sciences (l.naughton@wlv.ac.uk)


## References:

Aughney, T., Roche, N., Langton, S., 2018. The Irish Bat Monitoring Programme 2015-2017. Irish Wildlife Manuals, No. 103, National Parks and Wildlife Service. Department of Culture, Heritage and the Gaeltacht, Ireland.

Bertran, M., Alsina-Pages, R., Tena, E., 2019. Pipistrellus pipistrellus and pipistrellus pygmaeus in the iberian peninsula: An annotated segmented dataset and a proof of concept of a classifier in a real environment. Applied Sciences 9. doi:10.3390/app9173467.

Catto, C., Coyte, A., Agate, J., Langton, S., 2003. Bats as Indicators of Environmental Quality R&D Technical Report E1-129/TR. Technical Report ISBN: 1 844 32251 3. Environment Agency. Rio House, Waterside Drive, Aztec West, Almondsbury Bristol BS32 4UD. N.B. This document was produced under R&D Project E1-129 by the Bat Conservation Trust.

Gascoyne, A., Lomas, W., 2025. First-of-its-kind AI model for bioacoustic detection using a lightweight associative memory Hopfield neural network. Ecological Informatics, vol. 91, pp 103382, (2025). https://doi.org/10.1016/j.ecoinf.2025.103382.

Lomas, W., Gascoyne, A., Dubreuil, C., Vaglio, S., Naughton, L. (2025). Lightweight Hopfield Neural Networks for Bioacoustic Detection and Call Monitoring of Captive Primates. In: Arai, K. (eds) Proceedings of the Future Technologies Conference (FTC) 2025, Volume 1. FTC 2025. Lecture Notes in Networks and Systems, vol 1675. Springer, Cham. https://doi.org/10.1007/978-3-032-07986-2_38

Russ, J., 2021. Bat calls of Britain and Europe. Bat Biology and Conservation, Pelagic Publishing, Exeter, England.


## Acknowledgements:

The authors would like to thank the [OpenBright Foundation](https://openbright.org.uk/) and trustee Elizabeth Molyneux for their support and funding, and the [University of Wolverhampton](https://www.wlv.ac.uk/) for the Invest to Grow PhD Studentship funding. We would also like to thank staff at [Dudley Zoo and Castle (UK)](https://www.dudleyzoo.org.uk/) for their expert insight and assistance, and Jessica Gwilliams for data collection. Finally, thanks also go to Catherine Povey ([Just Mammals Ltd](https://www.justmammals.co.uk/)) for her expertise and useful discussions.  


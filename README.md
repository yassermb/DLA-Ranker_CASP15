### Contents

- [Overview](#overview)
- [Requirements](#Requirements)
- [Tutorial](#Tutorial)
- [License](./LICENSE)

## Citation:

```
@article {Mohseni Behbahani2022.04.05.487134,
	author = {Mohseni Behbahani, Yasser and Crouzet, Simon and Laine, {\'E}lodie and Carbone, Alessandra},
	title = {Deep Local Analysis evaluates protein docking conformations with locally oriented cubes},
	elocation-id = {2022.04.05.487134},
	year = {2022},
	doi = {10.1101/2022.04.05.487134},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/04/06/2022.04.05.487134},
	eprint = {https://www.biorxiv.org/content/early/2022/04/06/2022.04.05.487134.full.pdf},
	journal = {bioRxiv}
}
```

## Overview
![](Images/method5.svg.png?raw=true "DLA-Ranker")

Deep Local Analysis (DLA)-Ranker is a deep learning framework applying 3D convolutions to a set of locally oriented cubes representing the protein interface. It explicitly considers the local geometry of
the interfacial residues along with their neighboring atoms and the regions of the interface with different solvent accessibility. DLA-Ranker identifies near-native conformations and discovers alternative interfaces from ensembles generated by molecular docking.

#### Features:

- Useful APIs for fast preprocessing of huge assembly of the complex conformations and classify them based on CAPRI criteria. 

- Representation of an interface as a set of locally oriented cubes.
   - *Atomic density map as a 3D gird.*
   - *Structure class based on solvant accessibility (Support, Core, Rim).*
   - *Information on Receptor and Ligand.*
   - *Information of interfacial residues.*

- Classification of docking conformations based on CAPRI criteria (Incorrect, Acceptable, Medium, High quality)

- Fast generation of cubes and and evaluation of interface.

- Training and testing 3D-CNN models.

- Support various per-score aggregation schemes.
   - *Considering only subset cubes for evaluation of interface.*
   - *Residues from Support or Core regions.*
   - *Residues from Core or Rim regions.*
   - *Selecting residues exclusively from the receptor or from the ligand.*

- Extraction of embeddings and the topology of the interface for graph representation learning.



## Requirements

#### Packages:

DLA-Ranker can be run on Linux, MacOS, and Windows. We recommend to use DLA-Ranker on the machines with GPU. It requires following packages:
- [FreeSASA](https://github.com/mittinatten/freesasa) or [NACCESS](http://www.bioinf.manchester.ac.uk/naccess/)
- [ProDy] (http://prody.csb.pitt.edu/) 
- Python version 3.7 or 3.8.
- Tensorflow version 2.2 or 2.3.
- Cuda-Toolkit
- Scikit-Learn, numpy pandas matplotlib lz4 and tqdm (conda install -c pytorch -c pyg -c conda-forge python=3.9 numpy pandas matplotlib tqdm pytorch pyg scikit-learn cuda-toolkit).

All-in-one: Run conda create --name dla-ranker --file dla-ranker.yml

- For requirements of InteractionGNN please visit its Readme.

## Tutorial
Place the ensemble of conformations in a directory (*e.g. 'conformations_directory'*) like below: 

```
Evaluation
|___conformations_directory
    |
    |___target complex 1
    |   |   Conformation 1
    |   |   Conformation 2
    |   |   ...
    |
    |___target complex 2
    |   |   Conformation 1
    |   |   Conformation 2
    |   |   ...
    |
    ..........
```
Specify the path to FreeSASA or NACCESS in ```lib/tools.py``` (```FREESASA_PATH``` or ```NACCESS_PATH```). The choice between FreeSASA or NACCESS can be specified in the ```lib/tools.py``` (default is FreeSASA) <br>
If you have 'Nvidia GPU' on your computer, or execute on 'Google COLAB', set ```FORCE_CPU = False``` in ```lib/tools.py```. Otherwise set ```FORCE_CPU = True``` (default is FORCE_CPU=True). <br>

Run evaluation.py from Evaluation directory. It processes all the target complexes and their conformations and produces a csv file 'predictions_SCR' for each target complex. Each row of the output file belongs to a conformation and it has 9 columns separated by 'tab':

Name of target complex and the conformation (`Conf`) <br>
Fold Id (`Fold`) <br>
Score of each residue (`Scores`) <br>
Region (SCR) of each residue (`Regions`) <br>
Global averaged score of the interface (`Score`) <br>
Processing time (`Time`) <br>
Class of the conformation (`Class`, 0:incorrect, 1: near-native) <br>
Partner (`RecLig`) <br>
Residue number (`ResNumber`; according to PDB) <br>

One can associate the Residues' numbers, regions, scores, and partner to evaluate the interface on a subset of interfacial residues.

#### Acknowledgement
We would like to thank Dr. Sergei Grudinin and his team for helping us with the initial source code of ```maps_generator``` and ```load_data.py```. See [Ornate](https://academic.oup.com/bioinformatics/article/35/18/3313/5341430?login=true).


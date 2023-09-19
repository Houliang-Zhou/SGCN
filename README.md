# Multi-Modal Diagnosis of Alzheimer's Disease using Interpretable Graph Convolutional Networks
 A preliminary implementation of SGCN. In our experimentation, SGCN learned the sparse regional importance probability to find signature regions of interest (ROIs), and the connective importance probability to reveal disease-specific brain network connections.

## Usage
### Setup
The whole implementation is built upon [PyTorch](https://pytorch.org) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)

**conda**
See the `environment.yml` for environment configuration. 
```bash
conda env create -f environment.yml
```
**PYG**

To install pyg library, [please refer to the document](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Dataset 
**ADNI**
We download this dataset from [here](https://adni.loni.usc.edu/data-samples/access-data/).
We treat multi-modal imaging scans as a brain graph.

### How to run classification?
Training and testing are integrated in file `main.py`. To run
```
python main.py 
```
You can also specify the learning hyperparameters to run
```
python main.py --disease_id 0 --epochs 200 --lr 0.001 --search --cuda 0
```

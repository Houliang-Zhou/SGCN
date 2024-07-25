# Multi-Modal Diagnosis of Alzheimer's Disease using Interpretable Graph Convolutional Networks
 A preliminary implementation of the multi-modal sparse interpretable GCN framework (SGCN) for the detection of Alzheimer's disease (AD). In our experimentation, SGCN learned the sparse regional importance probability to find signature regions of interest (ROIs), and the connective importance probability to reveal disease-specific brain network connections.

## Note
The latest version of implementation is coming.

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
The whole training framework is integrated in file `main.py`. To run
```
python main.py 
```
You can also specify the learning hyperparameters to run
```
python main.py --disease_id 0 --epochs 200 --lr 0.001 --search --cuda 0
```

## Citation
If you find the code and dataset useful, please cite our paper.
```latex
@inproceedings{zhou2022sparse,
  title={Sparse interpretation of graph convolutional networks for multi-modal diagnosis of alzheimer’s disease},
  author={Zhou, Houliang and Zhang, Yu and Chen, Brian Y and Shen, Li and He, Lifang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={469--478},
  year={2022},
  organization={Springer}
}
```

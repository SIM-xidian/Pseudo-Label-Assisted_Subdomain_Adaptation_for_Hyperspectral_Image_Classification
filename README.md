# PASDA
Code for "Pseudo-Label-Assisted Subdomain Adaptation for Hyperspectral Image Classification".<br>
Zhixi Feng, Shilin Tong, Shuyuan Yang, Xinyu Zhang, and Licheng Jiao<br>
Xidian University

# Requirements

```
torch==1.12.1
torchvision==0.13.1
torchaudio==0.12.1
scipy==1.11.3
matplotlib==3.8.1
scikit-learn==1.3.2
```

# 

# Datasets

You can download the hyperspectral datasets in mat format at: https://pan.baidu.com/s/1QVSSiKxxgvOcnjHYt3INPQ?pwd=pexg, and move the files to `./datasets` folder.

An example dataset folder has the following structure:

```
datasets
├── Pavia
│   ├── paviaU.mat
│   └── paviaU_gt_7.mat
│   ├── pavia.mat
│   └── pavia_gt_7.mat
├── Houston
│   ├── Houston13.mat
│   └── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
├── HyRANK
│   ├── Dioni.mat
│   └── Dioni_gt_out68.mat
│   ├── Loukia.mat
│   └── Loukia_gt_out68.mat
```

# Usage:

Take PASDA method on the pavia dataset as an example:

1. Import our project and run a terminal.
2. Put the dataset into the correct path.
3. Run the command: pip install -r requirements.txt
4. Run pavia.py. 

# Thanks:
Some of our code references the projects
https://github.com/Li-ZK/CLDA-2022

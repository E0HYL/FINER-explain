# FINER
This repository contains the code and data associated with our CCS'23 publication (camera-ready version coming soon). An extended version of the paper, including an appendix, can be found on [arXiv](https://arxiv.org/pdf/2308.05362.pdf).

<!--
 If you find this research helpful for your publications, please kindly cite: 
```
@inproceedings{he2018finer,
  title={FINER: Enhancing State-of-the-art Classifiers with Feature Attribution to Facilitate Security Analysis},
  author={He, Yiling and Lou, Jian and Qin, Zhan and Ren, Kui},
  booktitle={Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security},
  pages={},
  year={2023}
}
```
-->

## Setup

```shell
conda env create --name FINER --file finer.yml
conda activate FINER
```

## How to run

All scripts can be found in `test/`. To run the experiments, use 

```shell
python -m unittest test/test_damd.py
python -m unittest test/test_deepreflect.py
python -m unittest test/test_vuldeepecker.py
```

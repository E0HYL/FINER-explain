# FINER

Data ( `examples/*/_data/`) and Code (to be released) for paper

*FINER: Enhancing State-of-the-art Classifiers with Feature Attribution to Facilitate Security Analysis*

> Code will be available when our paper is published.

## Setup

```shell
conda env create --name FINER --file FINER.yaml
conda activate FINER
```

## How to run

All scripts can be found in `test/`. To run the experiments, use 

```shell
python -m unit test/test_damd.py
python -m unit test/test_deepreflect.py
python -m unit test/test_vuldeepecker.py
```

# FINER

Code and Data for our CCS'23 paper (camera-ready version coming soon).

## Setup

```shell
conda env create --name FINER --file FINER.yaml
conda activate FINER
```

## How to run

All scripts can be found in `test/`. To run the experiments, use 

```shell
python -m unittest test/test_damd.py
python -m unittest test/test_deepreflect.py
python -m unittest test/test_vuldeepecker.py
```

# BDT (& more) Tools for Testing Run 3 R(K)

## Setting up Environment
You  can set up a computing environment with the necessary Python libraries using a Conda virtual environment `conda env create -f environment.yml`, or if you have a working python distribution, using pip `pip install --upgrade pip && pip install -r requirements.txt`.

## Running the scripts

### Preparing Data Inputs
    python prepare_inputs.py -m measure -j 10 -i <input dir> -o <output dir> -c data
### Preparing MC Inputs
    python prepare_inputs.py -m measure -j 1 -i <input dir> -o <output dir> -c rare

Same fore "rare", "jpsi", and "psi2s"

### BDT inference
    python bdt_inference.py -c bdt_cfg.yml [-v for verbose] [--debug for debug mode]

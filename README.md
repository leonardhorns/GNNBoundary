<h1>RE: GNNBoundary: Towards Explaining Graph Neural Networks through the Lens of Decision Boundaries</h1>

## How to use

### Model Checkpoints
* All model checkpoints can be found in `./ckpts`

### Datasets
* Here's the [link](https://drive.google.com/file/d/1O3IRF9mhL2KCCU1eVlCEdssaf6y-pq2h/view?usp=sharing) for downloading the processed datasets.
* After downloading the datasets zip, please `unzip` it in the root folder.
* To run the additional experiments on the REDDIT-BINARY dataset, download this dataset from [here](https://drive.google.com/file/d/1AE-2XH8QqMgQLrjtjcP0SZKrnR41Mwcc/view?usp=sharing) and unzip in the root folder like the other datasets.

### Environment
Codes in this repo have been tested on `python3.10` + `pytorch2.1` + `pyg2.5`.

To reproduce the exact python environment, please run:
```bash
conda create -n gnnboundary poetry jupyter
conda activate gnnboundary
poetry install
ipython kernel install --user --name=gnnboundary --display-name="GNNBoundary"
```

Note: In case poetry fails to install the dependencies, you can manually install them using `pip`:
```bash
pip install -r requirements.txt
```

### Reproducing Results
Once the environment is installed and the datasets are set up, our experiments can be easily reproduced by running the `run_experiments.py` script.

To reproduce the results for all datasets and models, run:
```bash
python run_experiments.py motif
python run_experiments.py collab
python run_experiments.py enzymes
python run_experiments.py binary_reddit
python run_experiments.py motif_gat
```

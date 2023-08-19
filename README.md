# AL4fairAI

## Setup
#### Requires python >= 3.10.4
1. Clone the repo
```
git clone https://github.com/ManishChandra12/AL4fairAI.git
```
2. cd to the project directory
```
cd AL4fairAI
```
3. Create the virtual environment
```
conda env create -f environment.yml
```
4. Activate the virtual environment
```
conda activate env
```

## Training and Evaluation
5. Run the following to train and evaluate on `CelebA` dataset
```
python -m src.main
```
Change `celebA` in `CONSTANTS.py` to `compas` or `eec` to train and evaluate on COMPAS, EEC datasets respectively. Refer to CONSTANTS.py for other hyperparameter settings.

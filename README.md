# Setup
The particular syntax for commands assumes Windows Powershell, but will work
with minor modifications for other command lines.

Clone this repository and enter the directory:

```bash
git clone https://github.com/MichaelHoltonPrice/lesciotto_et_al_sex_estimation_aafs_2024
cd lesciotto_et_al_sex_estimation_aafs_2024
```

Create and activate a virtual environment called mix (creating a virtual
environment is optional; you just need to ensure that you have installed all
the Python packages in requirements.txt):

```bash
python -m venv mix
Set-ExecutionPolicy Unrestricted -Scope Process
.\mix\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

To use a GPU, install pytorch with CUDA support (https://pytorch.org/get-started/locally/). This works on my current Windows machine:


```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# Create the results
If necessary, activate the virtual environment

```bash
Set-ExecutionPolicy Unrestricted -Scope Process
.\mix\Scripts\activate
```

Run the following two scripts to create the random forest and neural network cross validation results:

```bash
python create_random_forest_cross_validation_results.py
python create_basic_ann_cross_validation_results.py
```

This will create three files in the folders /outputs/random_forest and /outputs/basic_ann_ensemble:

- observer1_probabilities.csv
- observer2_probabilities.csv
- test_losses.json
- hyperparameters.json
    
The first two files contain the predicted probabilities for each observer and
observation, where the ordering of rows (observations) is identical to the
ordering in the input file, Pelvic-Sacral Pilot Data 2022.xlsx. These are
out-of-sample predictions using leave-one-out cross validation.

test_losses.json contains the mean test loss for each observer. The test metric
is cross-entropy loss, which equals the negative natural logarithm of the
probability of the known, true sex. The results will vary slightly each time
do to the random nature of the optimization algorithms.

hyperparameters.json contains the settings and hyperparameters for each the fit.
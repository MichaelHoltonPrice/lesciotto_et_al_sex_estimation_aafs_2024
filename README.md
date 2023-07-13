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

# Tests
If necessary, activate the virtual environment

```bash
Set-ExecutionPolicy Unrestricted -Scope Process
.\mix\Scripts\activate
```

Run some tests of the input parsing:

```bash
python -m unittest tests.TBD
```

# Create the results
If necessary, activate the virtual environment

```bash
Set-ExecutionPolicy Unrestricted -Scope Process
.\mix\Scripts\activate
```

Run the following script to create the random forest cross validation results:

```bash
python create_random_forest_cross_validation_results.py
```

This will create three files in the folder /outputs/random_forest:

- observer1_probabilities.csv
- observer2_probabilities.csv
- test_losses.json
    
The first two files contain the predicted probabilities for each observer and
observation, where the ordering of rows (observations) is identical to the
ordering in the input file, Pelvic-Sacral Pilot Data 2022.xlsx. These are
out-of-sample predictions using leave-one-out cross validation.
test_losses.json the mean test loss for each observer. As for all models that
we fit, the test metric is cross-entropy loss, which equals the negative
natural logarithm of the probability of the known, true sex. The results will
vary slightly each time do to the random nature of the eponymous random forest
fit.
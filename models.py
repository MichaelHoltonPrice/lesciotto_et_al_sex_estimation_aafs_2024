from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import numpy as np
from mixalot.datasets import MixedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
#from typing import Tuple, List, Dict

def fit_random_forest_wrapper(
        dataset_spec, 
        train_data,
        test_data,
        test_indices,
        hp
    ):
    """Train a random forest classifier and return test predictions.

    Given a dataset spec, training data, test data, test indices, and hyperparameters,
    this function trains a random forest classifier on the training data and 
    returns the trained classifier, summed loss on the test data, and a dictionary
    mapping study indices to predicted probabilities for the test data.

    Args:
        dataset_spec: DatasetSpec object describing the columns
        train_data: TrainData tuple with (X_categorical, X_ordinal, X_numerical)
        test_data: TestData tuple with (X_categorical, X_ordinal, X_numerical)
        test_indices: List of study indices for test data 
        hp: Dict with hyperparameters like {'num_estimators': 100}

    Returns:
        clf: Trained RandomForestClassifier
        fold_summed_test_loss: Total log loss on test data 
        study_index_to_prob: Dict mapping study indices to predicted probabilities
    
    """
    
    # Extract X and y arrays from training data
    # y has not yet been extracted from the following matrices. Create
    # a MixedDataset object that will accomplish the extraction for us.
    # Do so for both the training and test data.
    Xcat0, Xord0, Xnum0 = train_data
    mixed_dataset = MixedDataset(dataset_spec, Xcat0, Xord0, Xnum0)
    Xcat, Xord, Xnum, y = mixed_dataset.get_arrays()
    X = np.hstack([Xcat, Xord, Xnum])

    Xcat0_test, Xord0_test, Xnum0_test = test_data
    mixed_dataset_test = MixedDataset(dataset_spec, Xcat0_test, Xord0_test, Xnum0_test)
    Xcat_test, Xord_test, Xnum_test, y_test = mixed_dataset_test.get_arrays()

    # TODO: consider supporting imputation here

    # Train a random forest
    clf = RandomForestClassifier(n_estimators=hp['num_estimators'])
    clf.fit(X, y)

    # Predict the probabilities for test data
    X_test = np.hstack([Xcat_test, Xord_test, Xnum_test])
    y_pred_prob = clf.predict_proba(X_test)
    num_obs = y_pred_prob.shape[0]

    # Calculate the test loss for this fold (multiply by the number of
    # observations in this fold so that what we return is the total
    # test loss)
    #
    # We input the labels just in case all the values in y_test are the
    # same, which can lead to log_loss guessing incorrectly about how
    # y_test is indexed.
    fold_summed_test_loss = log_loss(y_test, y_pred_prob, labels=clf.classes_)*num_obs
    assert len(test_indices) == num_obs

    study_index_to_prob = dict()
    for i, original_index in enumerate(test_indices):
        values = y_pred_prob[i,:]
        study_index_to_prob[original_index] = values
    
    return clf, fold_summed_test_loss, study_index_to_prob

   
# TODO: update the number of viable models
def cross_validate(dataset_spec, train_test_folds, fold_test_indices, model_type, hp):
    """Cross validate a model by looping over the input folds.

    This function performs cross validation for either a random forest model 
    or an ensemble of basic feed forward artificial neural networks. It uses the 
    fit_random_forest_wrapper function for the random forest model. 

    Args:
        dataset_spec: DatasetSpec object describing the columns
        train_test_folds: List of tuples, each containing train and test data
        fold_test_indices: List of lists, each containing test indices for each fold
        model_type: String, type of the model for training ('random forest' or 'neural network')
        hp: Dict with hyperparameters like {'num_estimators': 100}

    Returns:
        overall_summed_test_loss: Total summed loss from all folds
        prob_matrix: Numpy array representing probabilities of predictions for all studies

    Raises:
        Exception: If the model_type is not recognized
    """
    
    # Initialize a dictionary that maps the study index to the predicted
    # probability. This will hold our eventual predictions.
    study_index_to_prob = {}
    
    # Initialize the overall loss (the summed loss, not the mean)
    overall_summed_test_loss = 0
    
    # Loop over folds. For each fold, we're going to train on the training data
    # and then make predictions on the test data.
    for fold_idx, (train_data, test_data) in enumerate(train_test_folds):
        test_indices = fold_test_indices[fold_idx]
        
        # Check the model type to determine the training and prediction process
        if model_type.lower() == 'random forest':
            # Use the wrapper function to fit the random forest model and obtain predictions
            _, fold_summed_test_loss, fold_study_index_to_prob =\
                fit_random_forest_wrapper(dataset_spec, train_data, test_data, test_indices, hp)
        else:
            raise Exception('Model type not supported yet')

        # Update the total loss and the prediction dictionary with the results from this fold
        overall_summed_test_loss += fold_summed_test_loss
        study_index_to_prob.update(fold_study_index_to_prob)

    # Create a numpy array from the dictionary study_index_to_prob. The keys are sorted to
    # ensure that the studies are always in the same order in the array.
    keys = np.array(sorted(study_index_to_prob.keys()))
    prob_matrix = np.array([study_index_to_prob[key] for key in keys])

    return  overall_summed_test_loss, prob_matrix


class InputTargetDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

# Cross validate a basic feed forward artificial neural network (ANN) model
# by looping over observers and folds. This network does not utilize masks to
# accommodate missing data, or any other other features included in some of the
# more sophisticated models below (TBD).
def cross_validate_basic_ann(dataset_spec,
                             obs1_folds,
                             obs2_folds,
                             fold_test_indices,
                             hidden_sizes,
                             dropout_prob,
                             num_models,
                             lr,
                             final_lr,
                             num_x_var,
                             batch_size,
                             epochs,
                             device):
    # Initialize dictionaries to store predicted test probabilities
    obs1_output_dict = {}
    obs2_output_dict = {}
    # Initialize the overall loss (the sumemd loss, not the mean)
    obs1_overall_test_loss = 0
    obs2_overall_test_loss = 0
    base_model_args = (num_x_var,
                       2,
                       hidden_sizes,
                       dropout_prob)
    # Loop over observers
    for obs_number in [1,2]:
        if obs_number == 1:
            folds = obs1_folds
        else:
            assert obs_number == 2
            folds = obs2_folds
        
        # Loop over folds
        for fold_idx, (train_data, test_data) in enumerate(folds):
            test_indices = fold_test_indices[fold_idx]
            # y has not yet been extracted from the following matrices. Create
            # a MixedDataset object that will accomplish the extraction for us.
            # Do so for both the training and test data.
            Xcat0, Xord0, Xnum0 = train_data
            mixed_dataset = MixedDataset(dataset_spec, Xcat0, Xord0, Xnum0)
            Xcat, Xord, Xnum, y = mixed_dataset.get_arrays()
            # need to start indexing from 0
            y = [k-1 for k in y]
            X = np.hstack([Xcat, Xord, Xnum])
            train_ds = InputTargetDataset(X,y)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            Xcat0_test, Xord0_test, Xnum0_test = test_data
            mixed_dataset_test = MixedDataset(dataset_spec, Xcat0_test, Xord0_test, Xnum0_test)
            Xcat_test, Xord_test, Xnum_test, y_test = mixed_dataset_test.get_arrays()
            # need to start indexing from 0
            y_test = [k-1 for k in y_test]
            X_test = np.hstack([Xcat_test, Xord_test, Xnum_test])
            test_ds = InputTargetDataset(X_test,y_test)
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

            # Train an ensemble of basic artificial neural network (ANN)
            ensemble = EnsembleTorchModel(num_models,
                                          lr,
                                          BasicAnn,
                                          *base_model_args,
                                          final_lr=final_lr)
            ensemble.train(train_dl, device, epochs, test_dl)

            # Predict the probabilities for test data
            num_obs = len(y_test)
            with torch.no_grad():
                test_input = torch.tensor(X_test, dtype=torch.float)
                y_pred_prob = ensemble.predict_prob(test_input, device)

            # Calculate the test loss for this fold (multiply by the number of
            # observations in this fold so that what we return is the total
            # test loss)
            #
            # We input the labels just in case all the values in y_test are the
            # same, which can lead to log_loss guessing incorrectly about how
            # y_test is indexed.
            #y_test = torch.tensor(y_test, device=device, dtype=torch.long)
            y_pred_prob = y_pred_prob.detach().cpu().numpy()
            fold_test_loss = log_loss(y_test, y_pred_prob, labels=[0,1])*num_obs
            if obs_number == 1:
                obs1_overall_test_loss += fold_test_loss
            else:
                assert obs_number == 2
                obs2_overall_test_loss += fold_test_loss

            assert len(test_indices) == num_obs
            for i, original_index in enumerate(test_indices):
                #values = y_pred_prob[i,:]
                values = y_pred_prob[i,:]
                if obs_number == 1:
                    obs1_output_dict[original_index] = values
                else:
                    assert obs_number == 2
                    obs2_output_dict[original_index] = values

    # Turn the dictionaries, which map the original indices onto probabilities,
    # into numpy arrays
    keys1 = np.array(sorted(obs1_output_dict.keys()))
    obs1_prob_matrix = np.array([obs1_output_dict[key] for key in keys1])

    keys2 = np.array(sorted(obs2_output_dict.keys()))
    obs2_prob_matrix = np.array([obs2_output_dict[key] for key in keys2])

    return  obs1_overall_test_loss, obs2_overall_test_loss, obs1_prob_matrix, obs2_prob_matrix


class BasicAnn(nn.Module):
    """
    Basic artificial neural network (ANN) model.

    This class represents a basic ANN model with an arbitrary number of hidden layers.
    The structure of the model includes an input layer, a sequence of alternating dense and dropout layers,
    and an output layer. The activation function applied after each dense layer is ReLU.

    Attributes:
        hidden_layers (nn.ModuleList): A sequence of alternating dense and dropout layers.
        output_layer (nn.Linear): The output layer of the model.
    """

    def __init__(self, num_x_var, num_cat, hidden_sizes, dropout_prob):
        """
        Initialize a new BasicAnn instance.

        Args:
            num_x_var (int): The number of input variables.
            num_cat (int): The number of output categories.
            hidden_sizes (list of int): A list of the sizes of the hidden layers.
            dropout_prob (float): The dropout probability for the dropout layers.
        """
        super(BasicAnn, self).__init__()

        self.hidden_layers = nn.ModuleList()
        
        # Define the input size to the first layer.
        input_size = num_x_var

        # Create the hidden layers
        for h, hidden_size in enumerate(hidden_sizes):
            # Dense layer.
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            # Dropout layer.
            self.hidden_layers.append(nn.Dropout(dropout_prob))

            # The output size of the current layer is the input size of the next layer.
            input_size = hidden_size

        # Define the output layer.
        self.output_layer = nn.Linear(input_size, num_cat)

    def forward(self, x):
        """
        Implement the forward pass.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor of the model.
        """
        # Pass through each hidden layer.
        for hidden_layer in self.hidden_layers:
            # Apply activation function after each dense layer.
            x = F.relu(hidden_layer(x))
            
        # Pass through the output layer. There is no activation function after the output layer.
        x = self.output_layer(x)

        return x


def train_one_epoch_for_basic_ann(model, dataloader, criterion, device, optimizer):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    total_obs = 0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Reset the gradients
        
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        batch_size = inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_obs += batch_size
        
    # Return the total loss divided by the number of observations
    return total_loss / total_obs

def test_for_basic_ann(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_obs = 0
    with torch.no_grad():  # No need to track gradients
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss

            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            total_obs += batch_size

    # Return the total loss divided by the number of observations
    return total_loss / total_obs

class EnsembleTorchModel:
    """
    A class that represents an ensemble of PyTorch models.

    This class handles the training of an ensemble of PyTorch models and prediction
    using the ensemble model. During prediction, the ensemble model outputs the 
    averaged probabilities from all the individual models.

    Attributes:
        models (list): List of PyTorch models forming the ensemble.
        lr (float): Learning rate for training the models.
        final_lr (float): Final learning rate for the LambdaLR scheduler.
    """

    def __init__(self,
                 num_models,
                 lr,
                 base_model_class,
                 *base_model_args,
                 final_lr=None,
                 **base_model_kwargs):
        """
        Initialize an EnsembleTorchModel instance.

        Args:
            num_models (int): The number of models in the ensemble.
            lr (float): The learning rate for training the models.
            base_model_class (type): The class of the models forming the ensemble.
            final_lr (float, optional): The final learning rate for the LambdaLR scheduler.
                                        If not provided, it will be the same as lr.
            *base_model_args: Positional arguments to be passed to base_model_class.
            **base_model_kwargs: Keyword arguments to be passed to base_model_class.
        """
        self.models = [base_model_class(*base_model_args, **base_model_kwargs)
                       for _ in range(num_models)]
        self.lr = lr
        self.final_lr = final_lr

    def train(self, train_dl, device, epochs, test_dl=None):
        """
        Train the ensemble model.

        Args:
            train_dl (torch.utils.data.DataLoader): The DataLoader for training data.
            device (torch.device): The device (CPU/GPU) to be used for training.
            epochs (int): The number of epochs for training.
            test_dl (torch.utils.data.DataLoader, optional): The DataLoader for test data.
                If provided, the function will also calculate, report, and return the test loss.

        Returns:
            float: The mean training loss for the ensemble model.
            float, optional: The mean test loss for the ensemble model. Only returned if test_dl is provided.
        """

        # Use cross entropy loss
        criterion = CrossEntropyLoss()

        # If a final learning rate is specified, use it
        initial_lr = self.lr
        if self.final_lr is not None:
            final_lr = self.final_lr
        else:
            final_lr = initial_lr
        # lambda1 is the learning rate decay, set to equal final_lr on the
        # final epoch
        lambda1 = lambda epoch: (final_lr / initial_lr) + (1 - epoch / epochs) * (1 - final_lr / initial_lr)

        # Initialize the training loss and (if necessary) the test loss
        ensemble_train_loss = 0.0
        total_train_obs = 0
        if test_dl is not None:
            ensemble_test_loss = 0.0
            total_test_obs = 0

        # Loop over models
        for i, model in enumerate(self.models, start=1):
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
            progress_bar = tqdm(range(epochs), desc=f"Training model #{i}")

            # Loop over epochs
            for epoch_num in progress_bar:
                epoch_loss = train_one_epoch_for_basic_ann(model, train_dl, criterion, device, optimizer)
                progress_bar.set_postfix({"epoch": epoch_num+1, "loss": epoch_loss})
            optimizer.step()
            scheduler.step()

            # Update training loss and number of observations
            summed_train_loss, num_train_obs = calc_summed_loss(model, train_dl, device)
            ensemble_train_loss += summed_train_loss
            mean_train_loss = summed_train_loss / num_train_obs
            total_train_obs += num_train_obs

            # If necessary, update training loss and number of observations
            if test_dl is not None:
                summed_test_loss, num_test_obs = calc_summed_loss(model, test_dl, device)
                ensemble_test_loss += summed_test_loss
                mean_test_loss = summed_test_loss / num_test_obs
                total_test_obs += num_test_obs
                mean_test_loss = summed_test_loss / num_test_obs
                print(f'Mean train loss = {mean_train_loss} and mean test loss = {mean_test_loss}')
            else:
                print(f'Mean train loss = {mean_train_loss}')
        
        if test_dl is not None:
            return ensemble_train_loss / total_train_obs, ensemble_test_loss / total_test_obs
        else:
            return ensemble_train_loss / total_train_obs

    def predict_prob(self, x, device):
        """
        Predict using the ensemble model.

        This method computes the averaged probabilities from all the individual models.
        The method does not return either of:
        - (1) unconstrained values, like the output of model.forward
        - (2) the log probabilities

        Args:
            x (torch.Tensor): The input tensor.
            device (torch.device): The device (CPU/GPU) to be used for prediction.

        Returns:
            torch.Tensor: The tensor of averaged probabilities.
        """
        all_probabilities = []
        for model in self.models:
            model.eval()
            probs = F.softmax(model(x.to(device)), dim=1)
            all_probabilities.append(probs)

        # Stack predictions to a tensor
        stacked_probabilities = torch.stack(all_probabilities)
        # Compute the averaged probabilities
        average_probabilities = torch.mean(stacked_probabilities, dim=0)

        return average_probabilities


def calc_summed_loss(model, dataloader, device):
    criterion = CrossEntropyLoss()
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_obs = 0
    with torch.no_grad():  # No need to track gradients
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss

            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            total_obs += batch_size

    # Return the total loss divided by the number of observations
    return total_loss, total_obs

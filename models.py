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
        hp: Dict with hyperparameters:
            'num _estimators': Number of estimators for the random forest

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

def fit_basic_ann_ensemble_wrapper(
        dataset_spec, 
        train_data,
        test_data,
        test_indices,
        hp,
        device
    ):
    """Train an ensemble of simple feed forward neural network classifiers and
    return test predictions.

    Given a dataset spec, training data, test data, test indices, and hyperparameters,
    this function trains an ensemble of neural networks on the training data and 
    returns the trained classifier, summed loss on the test data, and a dictionary
    mapping study indices to predicted probabilities for the test data.

    Args:
        dataset_spec: DatasetSpec object describing the columns
        train_data: TrainData tuple with (X_categorical, X_ordinal, X_numerical)
        test_data: TestData tuple with (X_categorical, X_ordinal, X_numerical)
        test_indices: List of study indices for test data 
        hp: Dict with hyperparameters:
            'batch_size': size of batches for model training
            'num_x_var': number of input features
            'hidden_sizes': list of number of neurons in each hidden layer
            'dropout_prob': probability of zeroing activations in dropout layers
            'num_models': number of models in the ensemble
            'lr': initial learning rate
            'final_lr': final learning rate for the exponential decay (use None for no decay)
            'epochs': number of complete passes over the training dataset
            'device': device for computation ('cpu' or 'cuda')

    Returns:
        ensemble: Trained EnsembleTorchModel object
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
    # need to start indexing from 0
    y = [k-1 for k in y]
    X = np.hstack([Xcat, Xord, Xnum])
    train_ds = InputTargetDataset(X,y)
    train_dl = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True)

    Xcat0_test, Xord0_test, Xnum0_test = test_data
    mixed_dataset_test = MixedDataset(dataset_spec, Xcat0_test, Xord0_test, Xnum0_test)
    Xcat_test, Xord_test, Xnum_test, y_test = mixed_dataset_test.get_arrays()
    # need to start indexing from 0
    y_test = [k-1 for k in y_test]
    X_test = np.hstack([Xcat_test, Xord_test, Xnum_test])
    test_ds = InputTargetDataset(X_test,y_test)
    test_dl = DataLoader(test_ds, batch_size=hp['batch_size'], shuffle=True)

    base_model_args = (hp['num_x_var'],
                       2, # there are two output categories (female and male)
                       hp['hidden_sizes'],
                       hp['dropout_prob'])
    # Train an ensemble of basic artificial neural network (ANN)
    ensemble = EnsembleTorchModel(hp['num_models'],
                                  hp['lr'],
                                  BasicAnn,
                                  *base_model_args,
                                  final_lr=hp['final_lr'])
    ensemble.train(train_dl, device, hp['epochs'], test_dl)

    # Predict the probabilities for test data
    num_obs = len(y_test)
    with torch.no_grad():
        test_input = torch.tensor(X_test, dtype=torch.float)
        y_pred_prob = ensemble.predict_prob(test_input, device).cpu().numpy()
    
    # Calculate the test loss for this fold (multiply by the number of
    # observations in this fold so that what we return is the total
    # test loss)
    #
    # We input the labels just in case all the values in y_test are the
    # same, which can lead to log_loss guessing incorrectly about how
    # y_test is indexed.
    fold_summed_test_loss = log_loss(y_test, y_pred_prob, labels=[0,1])*num_obs
    assert len(test_indices) == num_obs

    study_index_to_prob = dict()
    for i, original_index in enumerate(test_indices):
        values = y_pred_prob[i,:]
        study_index_to_prob[original_index] = values
    
    return ensemble, fold_summed_test_loss, study_index_to_prob

def train_cvae(train_dl, device, hp):
    """Train a Conditional Variational Autoencoder (CVAE).

    Args:
        train_dl (DataLoader): DataLoader for the training data.
        device (torch.device): The device to train the model on.
        hp (dict): Hyperparameters for training.

    Returns:
        CMixedVAE: Trained CVAE model.
    """
    # Initialize the model and move to device
    model = CMixedVAE(hp['cat_dims'],
                      hp['ord_dims'],
                      hp['num_dim'],
                      hp['latent_dim'],
                      hp['interior_dim'],
                      hp['beta'],
                      hp['dropout_prob']).to(device)

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])

    # Initialize the progress bar
    progress_bar = tqdm(range(hp['epochs']), desc="Training cVAE")

    # Training loop
    for epoch in progress_bar:
        model.train()
        train_loss = 0
        for i, (Xcat, Xord, Xnum, Mnum,
                cond_Xcat, cond_Xord, cond_Xnum, cond_Mnum,
                recon_mask_cat, recon_mask_ord, recon_mask_num) in enumerate(train_dl):
            # Move data to device (double check it is no longer necessary to move arrays to the device)
            #Xcat, Xord, Xnum, Mnum = Xcat.to(device), Xord.to(device), Xnum.to(device), Mnum.to(device)
            #cond_Xcat, cond_Xord, cond_Xnum, cond_Mnum = cond_Xcat.to(device), cond_Xord.to(device), cond_Xnum.to(device), cond_Mnum.to(device)

            # Forward pass
            recon_x, mu, logvar = model(Xcat, Xord, Xnum, Mnum,
                                        cond_Xcat, cond_Xord, cond_Xnum, cond_Mnum)

            # Compute loss
            recon_masks = (recon_mask_cat.to(device),
                           recon_mask_ord.to(device),
                           recon_mask_num.to(device))
            loss = model.vae_loss_function(recon_x, (Xcat, Xord, Xnum), mu, logvar, recon_masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Update the progress bar
        epoch_loss = train_loss / len(train_dl)
        progress_bar.set_postfix({"epoch": epoch+1, "loss": epoch_loss})

    return model

def fit_cvae_wrapper(dataset_spec,
                     train_data,
                     test_data,
                     test_indices,
                     hp,
                     device):
    """Train a conditional variational autoencoder (CVAE)

    TODO: finish documentation

   
    """
    dataset_spec.y_var = None
    Xcat, Xord, Xnum = train_data

    # The matrices are torch tensors, but ConditionedMixedDataset needs
    # numpy arrays.
    # TODO: consider supporting both torch tensors and numpy arrays in ConditionedMixedDataset
    Xcat = Xcat.cpu().numpy()
    Xord = Xord.cpu().numpy()
    Xnum = Xnum.cpu().numpy()

    train_ds = ConditionedMixedDataset(dataset_spec,
                                       Xcat,
                                       Xord,
                                       Xnum,
                                       hp['mask_prob'],
                                       hp['aug_mult'],
                                       device=device.type) # device.type is a string like 'cuda' or 'cpu'
    train_dl = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True)

    # For test data, we explicitly do our own conditioning below
    Xcat_test, Xord_test, Xnum_test = test_data
    Xcat_test = Xcat_test.cpu().numpy()
    Xord_test = Xord_test.cpu().numpy()
    Xnum_test = Xnum_test.cpu().numpy()

    test_ds = MixedDataset(dataset_spec,
                           Xcat_test,
                           Xord_test,
                           Xnum_test,
                           device=device.type)

    base_model_args = (hp['cat_dims'],
                       hp['ord_dims'],
                       hp['num_dim'],
                       hp['latent_dim'],
                       hp['interior_dim'],
                       hp['beta'],
                       hp['dropout_prob'])

    model = train_cvae(train_dl, device, hp)

    # Predict the probabilities for test data
    #num_obs = len(test_indices)
    # Condition on everything but Sex, which is what we want to predict
    ind_sex = dataset_spec.get_ordered_variables('categorical').index('Sex')
     
    with torch.no_grad():
        fold_summed_test_loss = 0
        study_index_to_prob = dict()
        for i, (Xcat_test, Xord_test, Xnum_test, Mnum_test) in enumerate(test_ds):
            known_sex = Xcat_test[ind_sex].item()
            assert known_sex in [1,2]
            # set sex to missing
            Xcat_test[ind_sex] = 0

            # The conditioned variables simply equal the input variables, now
            # that we've set sex to be missing
            Xcat_test, Xord_test, Xnum_test, Mnum_test =\
                Xcat_test.to(device), Xord_test.to(device), Xnum_test.to(device), Mnum_test.to(device)
            cond_Xcat_test = Xcat_test
            cond_Xord_test = Xord_test
            cond_Xnum_test = Xnum_test
            cond_Mnum_test = Mnum_test
            # TODO: do we need to make conditionined variables equal their value in the loss calc during training?
            recon_x, mu, logvar = model(Xcat_test, Xord_test, Xnum_test, Mnum_test,
                                        cond_Xcat_test, cond_Xord_test, cond_Xnum_test, cond_Mnum_test)
            pred_sex_logits = recon_x[0][ind_sex].cpu().numpy()
            pred_sex_prob = np.exp(pred_sex_logits)
            pred_sex_prob = pred_sex_prob / np.sum(pred_sex_prob)
            # TODO: we could ensemble models here
            fold_summed_test_loss += log_loss([known_sex], pred_sex_prob, labels=[1,2])

            original_index = test_indices[i]
            study_index_to_prob[original_index] = pred_sex_prob[0]
    
    return model, fold_summed_test_loss, study_index_to_prob


   
# TODO: update the number of viable models
def cross_validate(dataset_spec,
                   train_test_folds,
                   fold_test_indices,
                   hp,
                   device=None):
    """Cross validate a model by looping over the input folds.

    This function performs cross validation for either a random forest model 
    or an ensemble of basic feed forward artificial neural networks. It uses the 
    fit_random_forest_wrapper function for the random forest model. 

    Args:
        dataset_spec: DatasetSpec object describing the columns
        train_test_folds: List of tuples, each containing train and test data
        fold_test_indices: List of lists, each containing test indices for each fold
        model_type: String, type of the model for training ('random forest' or 'neural network')
        hp: Dictionary with hyperparameters, including the model type
        device (optional): The torch device. If not specified (and needed) the CPU is used

        device: The torch device 

    Returns:
        overall_summed_test_loss: Total summed loss from all folds
        prob_matrix: Numpy array representing probabilities of predictions for all studies

    Raises:
        Exception: If the model_type is not recognized
    """
    model_type = hp['model_type']
    
    if model_type.lower() in ['basic ann ensemble', 'cvae']:
        if device is None:
            device = torch.device('cpu')

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
                fit_random_forest_wrapper(dataset_spec,
                                          train_data,
                                          test_data,
                                          test_indices,
                                          hp)
        elif model_type.lower() == 'basic ann ensemble':
            # Use the wrapper function to fit the ensemble and obtain predictions
            _, fold_summed_test_loss, fold_study_index_to_prob =\
                fit_basic_ann_ensemble_wrapper(dataset_spec,
                                               train_data,
                                               test_data,
                                               test_indices,
                                               hp,
                                               device)
        elif model_type.lower() == 'cvae':
            # Use the wrapper function to fit the a CVAE
            _, fold_summed_test_loss, fold_study_index_to_prob =\
                fit_cvae_wrapper(dataset_spec,
                                 train_data,
                                 test_data,
                                 test_indices,
                                 hp,
                                 device)
        else:
            raise Exception(f'Unrecognized model_type = {model_type}')

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

class CMixedVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for mixed data.

    This class represents a Conditional Variational Autoencoder that is capable of handling mixed data. This includes
    categorical, ordinal, and numerical variables. The model also supports conditioning on one or more of these variables.
    Masked variables are supported as well.

    Attributes:
        cat_dims (list): List of dimensions for each categorical variable.
        ord_dims (list): List of dimensions for each ordinal variable.
        num_dim (int): Dimension of numerical variable.
        latent_dim (int): Dimension of latent space.
        interior_dim (int): Dimension of intermediate layer in encoder and decoder.
        beta (float): Weighting factor for the KL Divergence in the loss.
        dropout_prob (float): Probability of dropout.
    """

    def __init__(self, cat_dims, ord_dims, num_dim, latent_dim, interior_dim, beta, dropout_prob=0.5):
        """
        Initializes the CMixedVAE instance and sets up the necessary layers.

        Args:
            cat_dims (list): List of dimensions for each categorical variable.
            ord_dims (list): List of dimensions for each ordinal variable.
            num_dim (int): Dimension of numerical variable.
            latent_dim (int): Dimension of latent space.
            interior_dim (int): Dimension of intermediate layer in encoder and decoder.
            beta (float): Weighting factor for the KL Divergence in the loss.
            dropout_prob (float, optional): Probability of dropout. Defaults to 0.5.
        """
        super(CMixedVAE, self).__init__()

        self.cat_dims = cat_dims
        self.ord_dims = ord_dims
        self.num_dim = num_dim

        self.input_dim = len(cat_dims) + len(ord_dims) + 2 * num_dim
        self.beta = beta

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Encoder
        self.fc1 = nn.Linear(self.input_dim, interior_dim)
        self.fc21 = nn.Linear(interior_dim, latent_dim)
        self.fc22 = nn.Linear(interior_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim + self.input_dim, interior_dim)  # Add input dimension
        self.fc4 = nn.Linear(interior_dim, sum(cat_dims) + sum(ord_dims) + num_dim)

    def encode(self, x_m):
        """
        Encodes the input into a mean and log variance representation.

        Args:
            x_m (torch.Tensor): Input tensor containing stacked x-variable / mask input.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The means and log variance of the latent distribution.
        """
        h = self.dropout(torch.relu(self.fc1(x_m)))  # Concatenate condition variables
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        """
        Performs the reparameterization trick to sample from the latent space.

        Args:
            mu (torch.Tensor): The means of the latent distribution.
            logvar (torch.Tensor): The log variances of the latent distribution.

        Returns:
            torch.Tensor: The sampled latent variables.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond_x_m):
        """
        Decodes the latent variables back into the original space.

        Args:
            z (torch.Tensor): The latent variables.
            cond_x_m (torch.Tensor): Input tensor containing condition variables.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The reconstructed categorical, ordinal, and numerical variables.
        """
        h = self.dropout(torch.relu(self.fc3(torch.cat([z, cond_x_m], dim=-1))))  # Concatenate condition variables
        out = self.fc4(h)
        recon_cat = torch.split(out[:, :sum(self.cat_dims)], self.cat_dims, dim=1)
        recon_ord = torch.split(out[:, sum(self.cat_dims):sum(self.cat_dims) + sum(self.ord_dims)], self.ord_dims, dim=1)
        recon_num = out[:, -self.num_dim:]
        return recon_cat, recon_ord, recon_num

    def forward(self, x_cat, x_ord, x_num, m_num, cond_x_cat, cond_x_ord, cond_x_num, cond_m_num):
        """
        Defines the forward pass of the CVAE.

        Args:
            x_cat (torch.Tensor): The categorical input tensor.
            x_ord (torch.Tensor): The ordinal input tensor.
            x_num (torch.Tensor): The numerical input tensor.
            m_num (torch.Tensor): The numerical mask tensor.
            cond_x_cat (torch.Tensor): The categorical condition tensor.
            cond_x_ord (torch.Tensor): The ordinal condition tensor.
            cond_x_num (torch.Tensor): The numerical condition tensor.
            cond_m_num (torch.Tensor): The numerical condition mask tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The reconstructed categorical, ordinal, and numerical variables.
            tuple[torch.Tensor, torch.Tensor]: The means and log variances of the latent distribution.
        """

        # Check if the input tensors have a batch dimension; if not, add it
        if x_cat is not None and len(x_cat.shape) == 1:
            x_cat = x_cat.unsqueeze(0)
        if x_ord is not None and len(x_ord.shape) == 1:
            x_ord = x_ord.unsqueeze(0)
        if x_num is not None and len(x_num.shape) == 1:
            x_num = x_num.unsqueeze(0)
        if m_num is not None and len(m_num.shape) == 1:
            m_num = m_num.unsqueeze(0)
        if cond_x_cat is not None and len(cond_x_cat.shape) == 1:
            cond_x_cat = cond_x_cat.unsqueeze(0)
        if cond_x_ord is not None and len(cond_x_ord.shape) == 1:
            cond_x_ord = cond_x_ord.unsqueeze(0)
        if cond_x_num is not None and len(cond_x_num.shape) == 1:
            cond_x_num = cond_x_num.unsqueeze(0)
        if cond_m_num is not None and len(cond_m_num.shape) == 1:
            cond_m_num = cond_m_num.unsqueeze(0)

        x = torch.cat([x_cat, x_ord, x_num], dim=-1)
        m = torch.cat([m_num], dim=-1)
        cond_x = torch.cat([cond_x_cat, cond_x_ord, cond_x_num, cond_m_num], dim=-1)
        #cond_m = torch.cat([cond_m_num], dim=-1)
        mu, logvar = self.encode(torch.cat([x, m], dim=-1))
        z = self.reparameterize(mu, logvar)
        recon_cat, recon_ord, recon_num = self.decode(z, cond_x)
        return (recon_cat, recon_ord, recon_num), mu, logvar

    def vae_loss_function(self, recon_x, x, mu, logvar, recon_masks):
        """
        Defines the VAE loss function, including the reconstruction loss and the KL divergence.

        Args:
            recon_x (tuple): The reconstructed output of the VAE.
            x (tuple): The input tensors for the VAE (x_cat, x_ord, and x_num)
            mu (torch.Tensor): The means of the latent distribution.
            logvar (torch.Tensor): The log variances of the latent distribution.
            recon_masks (tuple): The binary reconstruction mask tensors where True indicates reconstruction required and
                                 False indicates no reconstruction.

        Returns:
            torch.Tensor: The total loss of the VAE.
        """
        recon_cat, recon_ord, recon_num = recon_x
        x_cat, x_ord, x_num = x
        recon_mask_cat, recon_mask_ord, recon_mask_num = recon_masks

        # Cross entropy for categorical and ordinal variables
        # For ordinal and categorical variables, a value of 0 indicates a mask
        # missing. For numerical variables, this is indicated by m_num. This
        # should already have been accounted for in the reconstruction masks.
        CE_ord = compute_masked_vae_cross_entropy(recon_ord, x_ord, recon_mask_ord)
        CE_cat = compute_masked_vae_cross_entropy(recon_cat, x_cat, recon_mask_cat)

        # Masked Mean Squared Error for numerical variables
        MSE_num = (recon_mask_num * (x_num - recon_num)**2).sum() / recon_mask_num.sum()

        # Overall loss
        loss = CE_cat + CE_ord + MSE_num

        # KL Divergence
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return loss + self.beta * KLD

def compute_masked_vae_cross_entropy(recon, x, recon_mask):
    """
    Compute masked cross entropy loss where recon_mask indicates what to reconstruct.

    Args:
        recon (torch.Tensor): The reconstructed output of the VAE.
        x (torch.Tensor): The input tensor for the VAE.
        recon_mask (torch.Tensor): The binary mask tensor where True indicates
                                   reconstruction required and False indicates
                                   no reconstruction.

    Returns:
        torch.Tensor: The masked cross entropy loss, averaged over unmasked
                      elements. None if all elements are masked.
    """
    batch_size = x.shape[0]
    already_init = False
    mask_count = 0
    CE = None
    # Let num_var be the number of variables and batch_size the batch size. Then:
    # len(recon) = num_var
    # recon[i].shape = (batch_size, num_cat_i)
    # x.shape = recon_mask.shape = batch_size x num_var
    for b in range(batch_size):
        for i in range(x.shape[1]):
            # If recon_mask is True for this variab le and observation, include
            # it in the loss
            if recon_mask[b, i]:
                known = x[b, i] - 1
                # TODO: consider making x be long to start with
                known = known.long()
                pred = recon[i][b, :]
                mask_count += 1
                if not already_init:
                    CE = torch.nn.functional.cross_entropy(pred, known)
                    already_init = True
                else:
                    CE += torch.nn.functional.cross_entropy(pred, known)

    if CE is None or mask_count == 0:
        return None

    CE = CE / mask_count
    return CE

def check_conditioning_vars(mask, conditioning_vars):
    """
    Check if any of the conditioning variables is masked.

    This function is used to ensure that the conditioning variables are not masked,
    as this would mean conditioning on missing values, which is not meaningful.
    
    Parameters:
    mask (torch.Tensor): The binary mask tensor where True indicates valid and False indicates masked.
    conditioning_vars (list): The list of indices of variables to condition on.

    Raises:
    ValueError: If any of the conditioning variables is masked.

    """
    for var in conditioning_vars:
        if mask[var] == 0:  # if variable is masked
            raise ValueError(f"Conditioning variable {var} is masked.")


class ConditionedMixedDataset(Dataset):
    """
    A PyTorch Dataset for mixed variable types (categorical, ordinal, numerical)
    with an artificial masking mechanism and random conditioning. This class
    constructs unmasked and conditioned versions of the input data. In the
    conditioned version, a random number of variables are masked.

    Attributes:
        mixed_dataset (MixedDataset): The base dataset containing mixed data
                                      that are potentially artificially masked
    """

    def __init__(self, dataset_spec, Xcat=None, Xord=None, Xnum=None,
                 mask_prob=0.0, aug_mult=1, device='cpu'):
        """
        Constructs the ConditionedMixedDataset.

        Args:
            dataset_spec (dict): The dataset specification dictionary.
            Xcat (np.ndarray): The categorical features array.
            Xord (np.ndarray): The ordinal features array.
            Xnum (np.ndarray): The numerical features array.
            mask_prob (float): The probability of masking an input variable.
            aug_mult (int): The augmentation multiplier.
        """
        assert dataset_spec.y_var is None
        self.mixed_dataset = MixedDataset(dataset_spec,
                                          Xcat=Xcat,
                                          Xord=Xord,
                                          Xnum=Xnum,
                                          mask_prob=mask_prob,
                                          aug_mult=aug_mult,
                                          require_input=True,
                                          device=device)

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.mixed_dataset)

    def __getitem__(self, idx):
        """
        Given an index, retrieves the corresponding item from the
        ConditionedMixedDataset. The item is a list consisting of corresponding
        elements (rows) from Xcat, Xord, Xnum, and their conditioned versions.
    
        Args:
            idx (int): The index of the item to be fetched.
    
        Returns:
            tuple: A tuple of length 11 containing Xcat, Xord, Xnum, and Mnum
                   along with their conditioned versions, and their corresponding 
                   reconstruction masks for categorical, ordinal, and numerical variables.
        """
        # Retrieve the item from mixed_dataset with artificial masking
        Xcat, Xord, Xnum, Mnum = self.mixed_dataset[idx]
    
        # Create conditioned copies
        cond_Xcat = Xcat.clone() if Xcat is not None else None
        cond_Xord = Xord.clone() if Xord is not None else None
        cond_Xnum = Xnum.clone() if Xnum is not None else None
        cond_Mnum = Mnum.clone() if Mnum is not None else None
    
        # Initialize reconstruction masks
        recon_mask_cat = torch.zeros_like(cond_Xcat, dtype=torch.bool) if Xcat is not None else None
        recon_mask_ord = torch.zeros_like(cond_Xord, dtype=torch.bool) if Xord is not None else None
        recon_mask_num = torch.zeros_like(cond_Mnum, dtype=torch.bool) if Xnum is not None else None
    
        # Combine datasets and conditioned versions for iteration
        datasets = [(Xcat, cond_Xcat, 'cat', recon_mask_cat), 
                    (Xord, cond_Xord, 'ord', recon_mask_ord), 
                    (Xnum, cond_Xnum, 'num', recon_mask_num)]
        masks = [None, None, Mnum]
    
        # Determine the non-masked variables across all variable types
        non_masked_indices = []
        variable_types = []
        for (X, _, type_id, _), mask in zip(datasets, masks):
            if X is not None:
                if type_id == 'num':  # For numerical variables, use mask to identify unmasked variables
                    X_non_masked_indices = mask
                else:  # For categorical and ordinal variables, a non-zero value indicates an unmasked variable
                    X_non_masked_indices = (X != 0)
    
                # Append indices of non-masked variables and their types
                #non_masked_indices.extend(np.where(X_non_masked_indices)[0].tolist())
                #non_masked_indices.extend(torch.nonzero(X_non_masked_indices).squeeze().tolist())
                non_masked_indices.extend(torch.where(X_non_masked_indices)[0].cpu().tolist())

                variable_types.extend([type_id] * torch.sum(X_non_masked_indices).item())
    
        num_not_masked = len(non_masked_indices)  # Count of non-masked variables across all variable types
    
        # TODO: add a switch to support never conditioning
        if num_not_masked > 0:
            # Randomly choose the number of variables to condition on.
            #num_condition = np.random.randint(1, num_not_masked+1)
    
            # Randomly choose which variables to condition on across all variable types
            #condition_indices = np.random.choice(range(num_not_masked),
            #                                     size=num_condition,
            #                                     replace=False)
            
            num_condition = torch.randint(1, num_not_masked+1, (1,)).item()
            condition_indices = torch.randperm(num_not_masked)[:num_condition].cpu().numpy()
    
            # Set all non-conditioned variables as masked in the copied dataset (for the recon mask)
            non_condition_indices = [index for index in range(num_not_masked) if index not in condition_indices]
            for non_condition_index in non_condition_indices:
                if variable_types[non_condition_index] == 'cat':
                    cond_Xcat[non_masked_indices[non_condition_index]] = 0
                    recon_mask_cat[non_masked_indices[non_condition_index]] = True
                elif variable_types[non_condition_index] == 'ord':
                    cond_Xord[non_masked_indices[non_condition_index]] = 0
                    recon_mask_ord[non_masked_indices[non_condition_index]] = True
                else:  # 'num'
                    assert variable_types[non_condition_index] == 'num'
                    cond_Xnum[non_masked_indices[non_condition_index]] = 0
                    cond_Mnum[non_masked_indices[non_condition_index]] = False
                    recon_mask_num[non_masked_indices[non_condition_index]] = True
    
        # Return original and conditioned datasets along with reconstruction masks
        return (Xcat, Xord, Xnum, Mnum, cond_Xcat, cond_Xord, cond_Xnum, cond_Mnum, 
                recon_mask_cat, recon_mask_ord, recon_mask_num)
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import numpy as np
from mixalot.datasets import MixedDataset
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.init as init

# Cross validate a random forest model by looping over observers and folds.
def cross_validate_rf(dataset_spec, obs1_folds, obs2_folds, fold_test_indices):
    # Initialize dictionaries to store predicted test probabilities
    obs1_output_dict = {}
    obs2_output_dict = {}
    # Initialize the overall loss (the sumemd loss, not the mean)
    obs1_overall_test_loss = 0
    obs2_overall_test_loss = 0
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
            X = np.hstack([Xcat, Xord, Xnum])

            Xcat0_test, Xord0_test, Xnum0_test = test_data
            mixed_dataset_test = MixedDataset(dataset_spec, Xcat0_test, Xord0_test, Xnum0_test)
            Xcat_test, Xord_test, Xnum_test, y_test = mixed_dataset_test.get_arrays()

            # TODO: consider supporting imputation here

            # Train a random forest
            clf = RandomForestClassifier(n_estimators=10000)
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
            fold_test_loss = log_loss(y_test, y_pred_prob, labels=clf.classes_)*num_obs
            if obs_number == 1:
                obs1_overall_test_loss += fold_test_loss
            else:
                assert obs_number == 2
                obs2_overall_test_loss += fold_test_loss

            assert len(test_indices) == num_obs
            for i, original_index in enumerate(test_indices):
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
                             init_scale,
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
                       dropout_prob,
                       init_scale)
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
            ensemble = EnsembleBasicAnn(num_models,
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

    def __init__(self, num_x_var, num_cat, hidden_sizes, dropout_prob, init_scale):
        super(BasicAnn, self).__init__()
        self.hidden_layers = nn.ModuleList()
        input_size = num_x_var
        
        
        self.init_scale = init_scale
    
        # Initialize weights with Xavier uniform method
        init_func = init.xavier_uniform_
        
        for h, hidden_size in enumerate(hidden_sizes):
          layer = nn.Linear(input_size, hidden_size)
          init_func(layer.weight) # initialize weights
          
          # Use smaller bias initialization  
          layer.bias.data.fill_(self.init_scale) 
          
          self.hidden_layers.append(layer)
          
          self.hidden_layers.append(nn.Dropout(dropout_prob))
          #if h+1 < len(hidden_sizes):
          #  self.hidden_layers.append(nn.Dropout(dropout_prob))
            
          input_size = hidden_size
          
        # Output layer
        self.output_layer = nn.Linear(input_size, num_cat)
        init_func(self.output_layer.weight)
        self.output_layer.bias.data.fill_(self.init_scale)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))  # apply activation function after each dense layer
            
        x = self.output_layer(x)  # no activation function after output layer
        return x



#class BasicAnn(nn.Module):
#    def __init__(self, num_x_var, num_cat, hidden_sizes, dropout_prob):
#        super(BasicAnn, self).__init__()
#        
#        self.hidden_layers = nn.ModuleList()
#        
#        input_size = num_x_var
#        for h, hidden_size in enumerate(hidden_sizes):
#            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
#            if h+1 < len(hidden_sizes):
#                # Do not use dropout on the final layer
#                self.hidden_layers.append(nn.Dropout(dropout_prob))
#            input_size = hidden_size  # the output size of the current layer is the input size of the next layer
#
#        # Output layer
#        self.output_layer = nn.Linear(input_size, num_cat)
#
#    def forward(self, x):
#        for hidden_layer in self.hidden_layers:
#            x = F.relu(hidden_layer(x))  # apply activation function after each dense layer
#            
#        x = self.output_layer(x)  # no activation function after output layer
#        return x

#def train_basic_ann():
#    # Train an ensemble of models to predict the category (Sex)
#    overall_test_loss = 0
#    criterion = CrossEntropyLoss()
#    base_model_args = (num_var,
#                       2,
#                       modelParam['hidden_sizes'],
#                       modelParam['dropout_prob'])
#    ensemble = EnsembleModel(num_models, lr, base_model_class, *base_model_args)
#    final_epoch_losses = ensemble.train(train_dl, device, epochs)
#    with torch.no_grad():
#        train_loss = np.mean(final_epoch_losses)
#        test_input = np.concatenate((Xtest, Mtest), axis=1)
#        test_input = torch.tensor(test_input, dtype=torch.float)
#        test_probs = ensemble.predict_prob(test_input, device)
#        test_logits = torch.log(test_probs)
#        targets = torch.tensor(ytest, dtype=torch.long).to(device)
#        loss = criterion(test_logits, targets)
#        num_test_obs += len(ytest)
#        overall_test_loss += loss.item() * len(ytest)

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

class EnsembleBasicAnn:
    def __init__(self,
                 num_models,
                 lr,
                 base_model_class,
                 *base_model_args,
                 final_lr=None,
                 **base_model_kwargs):
        self.models = [base_model_class(*base_model_args,
                                        **base_model_kwargs)
                                        for _ in range(num_models)]
        self.lr = lr
        self.final_lr = final_lr

    def train(self, train_dl, device, epochs, test_dl):
        criterion = CrossEntropyLoss()
        initial_lr = self.lr
        if self.final_lr is not None:
            final_lr = self.final_lr
        else:
            final_lr = initial_lr
        lambda1 = lambda epoch: (final_lr / initial_lr) + (1 - epoch / epochs) * (1 - final_lr / initial_lr)

        ensemble_train_loss = 0.0
        total_train_obs = 0
        for i, model in enumerate(self.models, start=1):
            model.to(device)
            #optimizer = Adam(model.parameters(), lr=self.lr)
            optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
            progress_bar = tqdm(range(epochs), desc=f"Training model #{i}")
            for epoch_num in progress_bar:
                epoch_loss = train_one_epoch_for_basic_ann(model, train_dl, criterion, device, optimizer)
                progress_bar.set_postfix({"epoch": epoch_num+1, "loss": epoch_loss})
            optimizer.step()
            scheduler.step()
            summed_train_loss, num_train_obs = calc_summed_loss(model, train_dl, device)
            summed_test_loss, num_test_obs = calc_summed_loss(model, test_dl, device)
            mean_train_loss = summed_train_loss / num_train_obs
            mean_test_loss = summed_test_loss / num_test_obs
            print(f'Mean train loss = {mean_train_loss} and mean test loss = {mean_test_loss}')
            ensemble_train_loss += summed_train_loss
            total_train_obs += num_train_obs
        
        # TODO: return the ensemble ploss here (based on averaging the training probabilities)
        return ensemble_train_loss / total_train_obs


    def predict_prob(self, x, device):
        # Predict ensembled probabilities. This does not return either of:
        # *not* (1) unconstrained values, like the output of model.forward
        # *not* (2) the log probabilities
        #
        # Ideally we might output unconstrained values as output by model.forward
        # for conceptual consistency, but that's not actually possible if we
        # average in the probability space, which I think is the correct thing
        # to do. The reason is that one cannot uniquely determine the x_i from
        # the p_i where the following relations hold:
        #
        # x_i is an unconstrained vector
        # p_i = SoftMax(x_i) = exp(x_i) / sum(exp(x))
        # z_i = log(p_i) = F.log_softmax
        #
        # Hence, we output probabilities
        all_probabilities = []
        for model in self.models:
            model.eval()
            probs = F.softmax(model(x.to(device)), dim=1)
            all_probabilities.append(probs)

        #all_probabilities = [F.softmax(model(x.to(device)), dim=1) for model in self.models]
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

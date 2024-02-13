import os
from sklearn.model_selection import KFold
from mixalot.datasets import load_mixed_data
import json
import pandas as pd
import numpy as np

def load_aafs_2024_data(num_folds, seed, remove_missing=False):
    spec_file_path = os.path.join('inputs', 'dataset_spec_aafs_2024.json')
    data_file_path = os.path.join('inputs', 'AAFS Data_MorphoPASSE and DSP2.xlsx')
    mixed_data, num_scalers = load_mixed_data(spec_file_path, data_file_path)

    # There are 204 total rows, where each row is for one specimin and one observer
    assert mixed_data.Xcat.shape == (204, 1)
    assert mixed_data.Xord.shape == (204, 3)
    assert mixed_data.Xnum.shape == (204, 10)
    
    # Load the raw data frame since we need to use the Observer and Specimen columns
    raw_df = pd.read_excel(data_file_path)

    if remove_missing:
        # Require no missing values for columns 7 to 19 of filtered_df (the input variables)
        filtered_df = raw_df.dropna(subset=raw_df.columns[7:19])
    else:
        filtered_df = raw_df

    observer_counts = filtered_df.groupby('Specimen')['Observer'].nunique()
    specimens_scored_by_both = observer_counts[observer_counts == 2].index.tolist()
    filtered_df = filtered_df[filtered_df['Specimen'].isin(specimens_scored_by_both)]

    # Assuming the order in mixed_data matches raw_df
    valid_specimens = filtered_df['Specimen'].unique()

    # Implementing the TODO: Iterating over valid_specimens to get the same ordering
    observer1_idx = []
    observer2_idx = []
    for specimen in valid_specimens:
        for observer in ['Alex', 'Kate']:
            # Get the matching entry for this specimen and observer
            row_index = filtered_df[(filtered_df['Specimen'] == specimen) & (filtered_df['Observer'] == observer)].index

            # Throw an error if we don't have exactly one match
            if len(row_index) != 1:
                raise Exception('Error: expected exactly one row for specimen {} and observer {}'.format(specimen, observer))
            
            if observer == 'Alex':
                observer1_idx.append(row_index[0])
            else:
                assert observer == 'Kate'
                observer2_idx.append(row_index[0])
            

    # Build the matrices for each observer
    obs1_data = (mixed_data.Xcat[observer1_idx], mixed_data.Xord[observer1_idx], mixed_data.Xnum[observer1_idx])
    obs2_data = (mixed_data.Xcat[observer2_idx], mixed_data.Xord[observer2_idx], mixed_data.Xnum[observer2_idx])

    # Partition the objects for each fold and each observer
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    obs1_folds = []
    obs2_folds = []
    fold_test_indices = []
    for train_idx, test_idx in kf.split(observer1_idx):  # Use observer1_idx for splitting
        # Organizing fold data for observer 1
        train1 = tuple(data[train_idx] for data in obs1_data)
        test1 = tuple(data[test_idx] for data in obs1_data)
        obs1_folds.append((train1, test1))
        
        # Organizing fold data for observer 2, matching with observer 1's partitioning
        train2 = tuple(data[train_idx] for data in obs2_data)
        test2 = tuple(data[test_idx] for data in obs2_data)
        obs2_folds.append((train2, test2))
        
        fold_test_indices.append(test_idx)

    # It's probably best to support specification of the y-variable in the json
    dataset_spec = mixed_data.dataset_spec
    dataset_spec.y_var = 'Sex'  # Ensuring y_var is correctly set
    return obs1_data, obs2_data, num_scalers, obs1_folds, obs2_folds, fold_test_indices, dataset_spec, valid_specimens

def load_innominate_data(num_folds, seed):
    spec_file_path = os.path.join('inputs', 'dataset_spec_innominate.json')
    data_file_path = os.path.join('inputs', 'Pelvic-Sacral Pilot Data 2022.xlsx')
    mixed_data, num_scalers = load_mixed_data(spec_file_path, data_file_path)

    # The first 35 observations are for observer 1 and the second 35 are for
    # N is the number of observations per observer.
    N = 35
    assert mixed_data.Xcat.shape == (2*N, 3)
    assert mixed_data.Xord.shape == (2*N, 5)
    assert mixed_data.Xnum.shape == (2*N, 9)
        
    observer1_idx = [n for n in range(N)]
    observer2_idx = [n + N for n in observer1_idx]

    # subset Xcat, Xord, and Xnum based on observer
    obs1_data = (mixed_data.Xcat[observer1_idx],
                 mixed_data.Xord[observer1_idx],
                 mixed_data.Xnum[observer1_idx])
    obs2_data = (mixed_data.Xcat[observer2_idx],
                 mixed_data.Xord[observer2_idx],
                 mixed_data.Xnum[observer2_idx])

    # Partition the objects for each fold and each observer
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    obs1_folds = []
    obs2_folds = []
    fold_test_indices = []
    N = len(obs1_data[0])
    for train_idx, test_idx in kf.split(range(N)):
        Xcat, Xord, Xnum = obs1_data
        train1 = (Xcat[train_idx], Xord[train_idx], Xnum[train_idx])
        test1 = (Xcat[test_idx], Xord[test_idx], Xnum[test_idx])
        fold1 = (train1, test1)
        obs1_folds.append(fold1)

        Xcat, Xord, Xnum = obs2_data
        train2 = (Xcat[train_idx], Xord[train_idx], Xnum[train_idx])
        test2 = (Xcat[test_idx], Xord[test_idx], Xnum[test_idx])
        fold2 = (train2, test2)
        obs2_folds.append(fold2)

        fold_test_indices.append(test_idx)

    # TODO: it's probably best to support specification of the y-variable in
    #       the json
    dataset_spec = mixed_data.dataset_spec
    dataset_spec.y_var = 'Sex'
    return obs1_data, obs2_data, num_scalers, obs1_folds, obs2_folds, fold_test_indices, dataset_spec


def save_cross_validation_results(output_folder,
                                  obs1_overall_test_loss,
                                  obs1_prob_matrix,
                                  obs2_overall_test_loss,
                                  obs2_prob_matrix,
                                  y_data,
                                  specimen_ids,
                                  hp):
    # Calculate and check the number of observations
    N = obs1_prob_matrix.shape[0]
    assert N == 92 # we only use operations scored by both observers
    assert obs2_prob_matrix.shape[0] == N

    # Convert y_data from a torch tensor to numpy array
    y_data = y_data.numpy()

    # Calculate loss vectors
    obs1_loss_vect = [-np.log(obs1_prob_matrix[n, y_data[n]-1]) for n in range(N)]
    obs2_loss_vect = [-np.log(obs2_prob_matrix[n, y_data[n]-1]) for n in range(N)]

    # Create output dataframes
    obs1_output_df = pd.DataFrame(obs1_prob_matrix, columns=['prob_female', 'prob_male'])
    obs1_output_df['cross_entropy_loss'] = obs1_loss_vect
    obs1_output_df['known_sex'] = ['Female' if v == 1 else 'Male' for v in y_data]
    obs1_output_df['specimen_id'] = specimen_ids

    obs2_output_df = pd.DataFrame(obs2_prob_matrix, columns=['prob_female', 'prob_male'])
    obs2_output_df['cross_entropy_loss'] = obs2_loss_vect
    obs2_output_df['known_sex'] = ['Female' if v == 1 else 'Male' for v in y_data]
    obs2_output_df['specimen_id'] = specimen_ids

    # Create folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Write dataframes to csv 
    obs1_output_df.to_csv(os.path.join(output_folder, 'observer1_probabilities.csv'), index=False)
    obs2_output_df.to_csv(os.path.join(output_folder, 'observer2_probabilities.csv'), index=False)

    # Write hyperparameters to a json file
    with open(os.path.join(output_folder, 'hyperparameters.json'), 'w') as f:
        json.dump(hp, f)

    # Compute mean test losses
    obs1_mean_test_loss = obs1_overall_test_loss / N
    obs2_mean_test_loss = obs2_overall_test_loss / N

    # Write the mean test losses to a json file
    test_losses = {'obs1_mean_test_loss': obs1_mean_test_loss,
                   'obs2_mean_test_loss': obs2_mean_test_loss}
    
    print('test_losses:')
    print(test_losses)
    with open(os.path.join(output_folder, 'test_losses.json'), 'w') as f:
        json.dump(test_losses, f)

    return obs1_mean_test_loss, obs2_mean_test_loss
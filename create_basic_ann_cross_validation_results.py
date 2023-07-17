import numpy as np
import os
from sexest import load_innominate_data
import json
from models import cross_validate
import torch

def main():
    obs1_data, obs2_data, num_scalers,\
        obs1_folds, obs2_folds, fold_test_indices, dataset_spec =\
            load_innominate_data(35, 884683)

    # If possible, use the GPU. Otherwise, use the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define hyperparameters
    # num_x_var could be determined from the other inputs
    hp = {'batch_size': 1024,
          'hidden_sizes': [20],
          'dropout_prob': 0.93,
          'num_models': 20,
          'lr': 1e-3,
          'final_lr': 1e-3,
          'num_x_var': 2 + 5 + 9,
          'epochs': 10000,
          'device': device}
 
    obs1_overall_test_loss, obs1_prob_matrix =\
        cross_validate(dataset_spec, obs1_folds, fold_test_indices, 'basic ann ensemble', hp)
    obs2_overall_test_loss, obs2_prob_matrix =\
        cross_validate(dataset_spec, obs2_folds, fold_test_indices, 'basic ann ensemble', hp)
    
    N = obs1_prob_matrix.shape[0]
    assert N == 35
    assert obs2_prob_matrix.shape[0] == N

    # Folder path 
    output_folder = os.path.join('outputs', 'random_forest')
    
    # Create folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Write numpy arrays to csv 
    column_names = ['Female', 'Male']
    header = ','.join(column_names)
    np.savetxt(os.path.join(output_folder, 'observer1_probabilities.csv'),
               obs1_prob_matrix,
               header=header)
    np.savetxt(os.path.join(output_folder, 'observer2_probabilities.csv'),
               obs2_prob_matrix,
               header=header)

    print('Mean test losses for observers 1 and 2, respectively:')
    obs1_mean_test_loss = obs1_overall_test_loss / N
    obs2_mean_test_loss = obs2_overall_test_loss / N
    print(obs1_mean_test_loss)
    print(obs2_mean_test_loss)

    # Write the mean test losses to a json file
    test_losses = {'obs1_mean_test_loss': obs1_mean_test_loss,
                   'obs2_mean_test_loss': obs2_mean_test_loss}
    
    with open(os.path.join(output_folder, 'test_losses.json'), 'w') as f:
        json.dump(test_losses, f)

 
if __name__ == '__main__':
    main()
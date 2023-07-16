import numpy as np
import pandas as pd
import os
from models import cross_validate_basic_ann
from sexest import load_innominate_data
import json
import torch
from mixalot.datasets import MixedDataset

def main():
    obs1_data, obs2_data, num_scalers,\
        obs1_folds, obs2_folds, fold_test_indices, dataset_spec =\
            load_innominate_data(35, 884683)
            #load_innominate_data(5, 884683)
    batch_size = 1024
    hidden_sizes = [20]
    dropout_prob = 0.93
    num_models = 2
    lr = 1e-3
    final_lr = 1e-3
    # num_x_var could be determined from the other inputs
    num_x_var = 2 + 5 + 9
    epochs = 10000
    #epochs = 1000
    # If possible, use the GPU. Otherwise, use the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    obs1_overall_test_loss, obs2_overall_test_loss, obs1_prob_matrix, obs2_prob_matrix=\
        cross_validate_basic_ann(dataset_spec,
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
                                 device)

    
    N = obs1_prob_matrix.shape[0]
    y_data = MixedDataset(dataset_spec, obs1_data[0], obs1_data[1], obs1_data[2]).y_data
    assert np.all(MixedDataset(dataset_spec, obs2_data[0], obs2_data[1], obs2_data[2]).y_data == y_data)
    obs1_loss_vect = [-np.log(obs1_prob_matrix[n, y_data[n]-1]) for n in range(N)]
    obs2_loss_vect = [-np.log(obs2_prob_matrix[n, y_data[n]-1]) for n in range(N)]

    obs1_output_df = pd.DataFrame(obs1_prob_matrix, columns=['prob_female', 'prob_male'])
    obs1_output_df['cross_entropy_loss'] = obs1_loss_vect
    obs1_output_df['known_sex'] = ['Female' if v == 1 else 'Male' for v in y_data]

    obs2_output_df = pd.DataFrame(obs2_prob_matrix, columns=['prob_female', 'prob_male'])
    obs2_output_df['cross_entropy_loss'] = obs2_loss_vect
    obs2_output_df['known_sex'] = ['Female' if v == 1 else 'Male' for v in y_data]


    # Folder path 
    output_folder = os.path.join('outputs', 'basic_ann')
    
    # Create folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Write numpy arrays to csv 
    obs1_output_df.to_csv(os.path.join(output_folder, 'observer1_probabilities.csv'),
                          index=False)
    obs2_output_df.to_csv(os.path.join(output_folder, 'observer2_probabilities.csv'),
                          index=False)

    print('Mean test losses for observers 1 and 2, respectively:')
    obs1_mean_test_loss = obs1_overall_test_loss / N
    obs2_mean_test_loss = obs2_overall_test_loss / N
    print(obs1_mean_test_loss)
    print(obs2_mean_test_loss)
    print(np.mean(obs1_loss_vect))
    print(np.mean(obs2_loss_vect))

    # Write the mean test losses to a json file
    test_losses = {'obs1_mean_test_loss': obs1_mean_test_loss,
                   'obs2_mean_test_loss': obs2_mean_test_loss}
    
    with open(os.path.join(output_folder, 'test_losses.json'), 'w') as f:
        json.dump(test_losses, f)

 
if __name__ == '__main__':
    main()
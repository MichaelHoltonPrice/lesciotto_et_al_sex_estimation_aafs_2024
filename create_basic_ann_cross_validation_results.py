import numpy as np
import os
from sexest import load_innominate_data
from models import cross_validate
import torch
from sexest import save_cross_validation_results
from mixalot.datasets import MixedDataset

def main():
    # Define hyperparameters
    # num_x_var could be determined from the other inputs
    hp = {'model_type': 'basic ann ensemble',
          'batch_size': 1024,
          'hidden_sizes': [20],
          'dropout_prob': 0.93,
          'num_models': 20,
          'lr': 1e-3,
          'final_lr': 1e-3,
          'num_x_var': 2 + 5 + 9,
          'epochs': 10000,
          'num_folds': 35,
          'fold_seed': 884683}
 
    obs1_data, obs2_data, num_scalers,\
        obs1_folds, obs2_folds, fold_test_indices, dataset_spec =\
            load_innominate_data(hp['num_folds'], hp['fold_seed'])

    # If possible, use the GPU. Otherwise, use the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    obs1_overall_test_loss, obs1_prob_matrix =\
        cross_validate(dataset_spec, obs1_folds,
                       fold_test_indices,
                       hp,
                       device=device)
    obs2_overall_test_loss, obs2_prob_matrix =\
        cross_validate(dataset_spec,
                       obs2_folds,
                       fold_test_indices,
                       hp,
                       device=device)
    
    y_data = MixedDataset(dataset_spec, obs1_data[0], obs1_data[1], obs1_data[2]).y_data
    assert np.all(MixedDataset(dataset_spec, obs2_data[0], obs2_data[1], obs2_data[2]).y_data == y_data)
    
    save_cross_validation_results(os.path.join('outputs', 'basic_ann_ensemble'),
                                  obs1_overall_test_loss,
                                  obs1_prob_matrix,
                                  obs2_overall_test_loss,
                                  obs2_prob_matrix,
                                  y_data,
                                  hp)
 
if __name__ == '__main__':
    main()
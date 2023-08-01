import numpy as np
import os
from sexest import load_innominate_data
from models import cross_validate
import torch
from sexest import save_cross_validation_results
from mixalot.datasets import MixedDataset

def main():

    num_folds = 7
    #num_folds = 35
    fold_seed = 884683
    obs1_data, obs2_data, num_scalers,\
        obs1_folds, obs2_folds, fold_test_indices, dataset_spec =\
            load_innominate_data(num_folds, fold_seed)

    # Define hyperparameters
    cat_var_names = dataset_spec.get_ordered_variables('categorical')
    cat_dims = [len(dataset_spec.get_var_spec(var_name).categorical_mapping)\
                for var_name in cat_var_names]
    ord_var_names = dataset_spec.get_ordered_variables('ordinal')
    ord_dims = [len(dataset_spec.get_var_spec(var_name).categorical_mapping)\
                for var_name in ord_var_names]
    num_dim = len(dataset_spec.get_ordered_variables('numerical'))


    # TODO: place dataloading on device to speed up training
    hp = {'model_type': 'cvae',
          'batch_size': 1024,
          'cat_dims': cat_dims,
          'ord_dims': ord_dims,
          'num_dim': num_dim,
          'latent_dim': 8,
          'interior_dim': 64,
          'mask_prob': 0.0,
          'aug_mult': 1,
          'beta': 1.0,
          'dropout_prob': 0.5,
          'lr': 1e-3,
          'final_lr': 1e-4,
          #'epochs': 10,
          'epochs': 1000,
          'num_folds': num_folds,
          'fold_seed': fold_seed}
 
    # If possible, use the GPU. Otherwise, use the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    obs1_overall_test_loss, obs1_prob_matrix =\
        cross_validate(dataset_spec,
                       obs1_folds,
                       fold_test_indices,
                       hp,
                       device=device)
    obs2_overall_test_loss, obs2_prob_matrix =\
        cross_validate(dataset_spec,
                       obs2_folds,
                       fold_test_indices,
                       hp,
                       device=device)
    
    # Add back the y-variable in dataset_spec
    dataset_spec.y_var = 'Sex'
    y_data = MixedDataset(dataset_spec, obs1_data[0], obs1_data[1], obs1_data[2]).y_data
    assert np.all(MixedDataset(dataset_spec, obs2_data[0], obs2_data[1], obs2_data[2]).y_data == y_data)
    
    save_cross_validation_results(os.path.join('outputs', 'cvae'),
                                  obs1_overall_test_loss,
                                  obs1_prob_matrix,
                                  obs2_overall_test_loss,
                                  obs2_prob_matrix,
                                  y_data,
                                  hp)
 
if __name__ == '__main__':
    main()
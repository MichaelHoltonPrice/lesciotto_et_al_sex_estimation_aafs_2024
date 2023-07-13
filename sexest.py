import os
from sklearn.model_selection import KFold
from mixalot.datasets import load_mixed_data

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
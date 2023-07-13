import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import numpy as np
from mixalot.datasets import MixedDataset

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
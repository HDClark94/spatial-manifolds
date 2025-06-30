import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

def cross_validate_decoder_time(tcs, 
                                true_position, 
                                trial_numbers, 
                                tns_to_decode, 
                                tns_to_train,
                                tl,
                                bs,
                                train=0.9, 
                                n=50,
                                verbose=True):
    """
    tcs: dictionary of tuning curves, each entry is a 2d numpy array of length (n cells, n spatial bins)
    true_posiition: numpy array of position bin centres of length (n spatial bins)
    trial_numbers: numpy array of trial numbers of length (n spatial bins)
    tns_to_decode: numpy array of trial numbers in which to decode with of length (n trials to decode with)
    tns_to_train: numpy array of trial numbers in which to train with of length (n trials to train with)
    tl: track length in cm, scalar
    train: proportion of full set to train the decoder with
    test: proportion of full set to test the decoder with
    train and test must equal to 1
    """

    # create a train test regime
    # take 10% of trials to test with, save the estimates
    # repeat until all trials have n repeats

    predictions = [[ [] for _ in range(n)] for _ in range(len(tns_to_decode))]
    errors = [[ [] for _ in range(n)] for _ in range(len(tns_to_decode))]
    n_counter = np.zeros((len(tns_to_decode)), dtype=np.int64)

    while np.sum(n_counter)<(len(tns_to_decode)*n):
        #print(f'{np.sum(np.isnan(errors))}\{errors.shape[0]*errors.shape[1]*errors.shape[2]}')
        if verbose:
            print(f'{np.round(100*np.sum(n_counter)/(len(tns_to_decode)*n), decimals=1)}%')

        # select x% to train iteration with
        train_tns = np.random.choice(tns_to_train, size=int(train*len(tns_to_train)), replace=False)
        # remove the training trials from the test trials
        test_tns = np.setdiff1d(tns_to_decode, train_tns) 

        error, true_tns, true_ps, pred_ps = circular_decoder(tcs, 
                                                             true_position, 
                                                             trial_numbers, 
                                                             train_tns, 
                                                             test_tns,
                                                             tl, 
                                                            )
        
        for tn in np.unique(true_tns):
            tn_pred = pred_ps[true_tns == tn]
            tn_error = error[true_tns == tn]
            tn_i = np.where(tns_to_decode == tn)[0][0]
            tn_n = n_counter[tn_i]
            if tn_n<n:
                predictions[tn_i][tn_n] = tn_pred
                errors[tn_i][tn_n] = tn_error
                n_counter[tn_i]+=1

    # return the mean decoder error and the last iteration of prediction and true position for plotting
    return predictions, errors




def cross_validate_decoder(tcs, 
                           true_position, 
                           trial_numbers, 
                           tns_to_decode, 
                           tns_to_train,
                           tl,
                           bs,
                           train=0.9, 
                           n=50,
                           verbose=False):
    """
    tcs: dictionary of tuning curves, each entry is a 2d numpy array of length (n cells, n spatial bins)
    true_posiition: numpy array of position bin centres of length (n spatial bins)
    trial_numbers: numpy array of trial numbers of length (n spatial bins)
    tns_to_decode: numpy array of trial numbers in which to decode with of length (n trials to decode with)
    tns_to_train: numpy array of trial numbers in which to train with of length (n trials to train with)
    tl: track length in cm, scalar
    train: proportion of full set to train the decoder with
    test: proportion of full set to test the decoder with
    train and test must equal to 1
    """

    # create a train test regime
    # take 10% of trials to test with, save the estimates
    # repeat until all trials have n repeats

    predictions = np.zeros((len(tns_to_decode), int(tl/bs), n)); predictions[:] = np.nan
    errors = np.zeros((len(tns_to_decode), int(tl/bs), n)); errors[:] = np.nan
    n_counter = np.zeros(len(tns_to_decode), dtype=np.int64)

    while np.sum(np.isnan(errors))>0:
        #print(f'{np.sum(np.isnan(errors))}\{errors.shape[0]*errors.shape[1]*errors.shape[2]}')
        if verbose:
            print(f'{np.round(100*np.sum(np.isnan(errors))/(errors.shape[0]*errors.shape[1]*errors.shape[2]), decimals=1)}%')

        # select 80% to train iteration with
        train_tns = np.random.choice(tns_to_train, size=int(train*len(tns_to_train)), replace=False)
        # remove the training trials from the test trials
        test_tns = np.setdiff1d(tns_to_decode, train_tns) 

        error, true_tns, true_ps, pred_ps = circular_decoder(tcs, 
                                                             true_position, 
                                                             trial_numbers, 
                                                             train_tns, 
                                                             test_tns,
                                                             tl, 
                                                            )
        
        for tn in np.unique(true_tns):
            tn_pred = pred_ps[true_tns == tn]
            tn_error = error[true_tns == tn]
            tn_i = np.where(tns_to_decode == tn)[0][0]
            tn_n = n_counter[tn_i]
            if tn_n<n:
                predictions[tn_i,:,tn_n] = tn_pred
                errors[tn_i,:,tn_n] = tn_error
                n_counter[tn_i]+=1

    # return the mean decoder error and the last iteration of prediction and true position for plotting
    return predictions, errors

def circular_decoder(tcs, 
                     true_position, 
                     trial_numbers, 
                     train_tns,
                     test_tns, 
                     tl,
                     ):
    """
    tcs: dictionary of tuning curves, each entry is a 2d numpy array of length (n cells, n spatial bins)
    true_posiition: numpy array of position bin centres of length (n spatial bins)
    trial_numbers: numpy array of trial numbers of length (n spatial bins)
    train_tns: numpy array of trial numbers in which to train decoder (n trials to train with)
    test_tns: numpy array of trial numbers in which to test decoder (n trials to test with)
    tl: track length in cm, scalar
    """
    assert len(np.intersect1d(train_tns, test_tns))==0, print(f'it looks like there are some trials in both train and test sets')
    
    true_angles = (true_position/tl) * 2 * np.pi -np.pi
    rates = np.vstack(list(tcs.values())).T

    # make train and test mask
    train_mask = np.isin(trial_numbers, train_tns)
    test_mask = np.isin(trial_numbers, test_tns)

    # Circular-linear coordinates
    cos_x = np.cos(true_angles)
    sin_x = np.sin(true_angles)

    # Split train and test
    rates_train, rates_test = rates[train_mask], rates[test_mask]
    cos_x_train, cos_x_test = cos_x[train_mask], cos_x[test_mask]
    sin_x_train, sin_x_test = sin_x[train_mask], sin_x[test_mask]

    # Train the decoder
    reg_cos = LinearRegression().fit(rates_train, cos_x_train)
    reg_sin = LinearRegression().fit(rates_train, sin_x_train)
    
    # Predict angles
    cos_x_pred = reg_cos.predict(rates_test)
    sin_x_pred = reg_sin.predict(rates_test)
 
    # Convert angles to predicted positions
    pred_angles = np.arctan2(sin_x_pred, cos_x_pred)
   
    # Convert the predictions back to position units
    pred_position = (pred_angles + np.pi) / (2 * np.pi) * tl
    
    # Evaluate the decoder  
    pred_angle_0to2pi = pred_angles+np.pi
    true_angles_0to2pi = true_angles+np.pi
    error_angle = np.min(np.stack([np.abs(pred_angle_0to2pi-true_angles_0to2pi[test_mask]),
                                   (2*np.pi)-np.abs(pred_angle_0to2pi-true_angles_0to2pi[test_mask])]), axis=0)
    error_cm = (error_angle / (2 * np.pi)) * tl

    return error_cm, trial_numbers[test_mask], true_position[test_mask], pred_position


def circular_nanmean(estimates, tl, axis=0):
    angles = (estimates/tl) * 2 * np.pi - np.pi
    cos = np.cos(angles)
    sin = np.sin(angles)
    mean_cos = np.nanmean(cos, axis=axis)
    mean_sin = np.nanmean(sin, axis=axis)
    mean_angles = np.arctan2(mean_sin, mean_cos)
    return (mean_angles + np.pi) / (2 * np.pi) * tl


def circular_nansem(estimates, tl, axis=0):
    angles = (estimates/tl) * 2 * np.pi - np.pi
    cos = np.cos(angles)
    sin = np.sin(angles)
    mean_cos = stats.sem(cos, axis=axis, nan_policy='omit')
    mean_sin = stats.sem(sin, axis=axis, nan_policy='omit')
    mean_angles = np.arctan2(mean_sin, mean_cos)
    return (mean_angles + np.pi) / (2 * np.pi) * tl
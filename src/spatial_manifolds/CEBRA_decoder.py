import numpy as np
from sklearn.linear_model import LinearRegression
from cebra import CEBRA
import cebra.integrations.plotly
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

def cross_validate_CEBRA_decoder(tcs, 
                           true_position, 
                           trial_numbers, 
                           tns_to_decode, 
                           tns_to_train,
                           tl,
                           bs,
                           train=0.5, 
                           n=2):
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

    for _ in range(n):
        np.random.shuffle(tns_to_decode)
        for test_tns in np.array_split(tns_to_decode, int(1/(1-train))):
            train_tns = np.setdiff1d(tns_to_train, test_tns) 
            error, true_tns, true_ps, pred_ps = CEBRA_decoder(tcs, 
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

            print(f' CEBRA decoding {100-(100*(np.sum(np.isnan(errors))/len(errors.flatten())))}% complete')

            

    # return the mean decoder error and the last iteration of prediction and true position for plotting
    return predictions, errors



def CEBRA_decoder(tcs, 
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
    
    max_iterations = 10000
    output_dimension = 10  # here, we set as a variable for hypothesis testing below.
    cebra_model = CEBRA(model_architecture='offset10-model',
                                batch_size=512,
                                learning_rate=3e-4,
                                temperature=10,
                                output_dimension=output_dimension,
                                max_iterations=max_iterations,
                                distance='cosine',
                                conditional='time_delta',
                                device='cuda_if_available',
                                verbose=True,
                                time_offsets=1)

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

    label_train = np.stack([sin_x_train, cos_x_train], axis=0).T

    cebra_model.fit(rates_train, label_train)
    cebra_train = cebra_model.transform(rates_train)
    cebra_test = cebra_model.transform(rates_test)

    pos_decoder = KNeighborsRegressor(36, metric='cosine')
    pos_decoder.fit(cebra_train, label_train)
    pos_pred = pos_decoder.predict(cebra_test)

    sin_x_pred = pos_pred[:,0]
    cos_x_pred = pos_pred[:,1]

    # Convert angles to predicted positions
    pred_angles = np.arctan2(sin_x_pred, cos_x_pred)
   
    # Convert the predictions back to position units
    pred_position = (pred_angles + np.pi) / (2 * np.pi) * tl
 
    # Evaluate the decoder  
    error_cm = pred_position - true_position[test_mask]
    error_cm[error_cm>(tl*0.75)] = tl-error_cm[error_cm>(tl*0.75)]
    error_cm[error_cm<(-tl*0.75)] = tl+error_cm[error_cm<(-tl*0.75)]

    return error_cm, trial_numbers[test_mask], true_position[test_mask], pred_position





def encode_1d_to_2d(positions, min_val=0, max_val=200):
    # Calculate the circumference
    circumference = max_val - min_val

    # Normalize positions to fall between 0 and 1
    normalized_positions = (positions - min_val) / circumference

    # Calculate 2D coordinates
    x_positions = np.cos(2 * np.pi * normalized_positions)
    y_positions = np.sin(2 * np.pi * normalized_positions)

    return np.array([x_positions, y_positions]).T


def decode_2d_to_1d(coordinates, min_val=0, max_val=200):
    # Calculate the circumference
    circumference = max_val - min_val

    # Transform 2D coordinates back into angles
    angles = np.arctan2(coordinates[:,1], coordinates[:,0])

    # Normalize angles to fall between 0 and 1
    normalized_angles = (angles % (2 * np.pi)) / (2 * np.pi)

    # Calculate original positions
    positions = normalized_angles * circumference + min_val
    return positions
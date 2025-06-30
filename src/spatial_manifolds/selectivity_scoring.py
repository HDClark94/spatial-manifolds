import numpy as np
from scipy.interpolate import interp1d

def curate_behavioural_mask(mask):
    found_true = False
    for i in range(len(mask)):
        if mask[i]:
            if found_true:
                mask[i] = False
            else:
                found_true = True
        else:
            found_true = False
    return mask

def get_lick_selectivity(session):
    trial_numbers = np.array(session['trial_number'])
    position = np.array(session['P'])

    position_times = np.array(session['P'].index)
    lick_times = np.array(session['lick'].index)
    lick_mask = np.array(session['lick'])
    lick_mask = curate_behavioural_mask(lick_mask)
    lick_times = np.insert(lick_times, 0, 0) # add a t=0
    lick_mask = np.insert(lick_mask, 0, 0) # assign this lick=False
    lick_times = np.insert(lick_times, -1, position_times[-1]) # add a t=last position time
    lick_mask = np.insert(lick_mask, -1, 0) # assign this lick=False
    itpe = interp1d(lick_times, lick_mask, kind='nearest')
    lick_mask = itpe(position_times).astype(bool)

    # calc stop selectivity
    # 10 cm before rz is included for anticipitory action
    stop_selectivities = []
    lick_sensitivities = []
    for tn in np.unique(trial_numbers):
        trial_context = session['trials'][session['trials']['trial_number'] == tn]['trial_context'].values[0]
        if trial_context == 'rz1':
            rz_bounds = (80, 110)
        elif trial_context == 'rz2':
            rz_bounds = (110, 140)

        tn_mask = trial_numbers == tn
        rz_mask = ((position>rz_bounds[0]) & (position<rz_bounds[1]))
        elsewhere_mask = ((position<rz_bounds[0]) | (position>rz_bounds[1]))
        
        n_licks_in_rz =     len(position[(tn_mask & lick_mask & rz_mask)])
        n_licks_elsewhere = len(position[(tn_mask & lick_mask & elsewhere_mask)])
        lick_sensitivity = (n_licks_in_rz - n_licks_elsewhere)/(n_licks_in_rz + n_licks_elsewhere + 1e-10)

        lick_sensitivities.append(lick_sensitivity)

    stop_selectivities = np.array(stop_selectivities)

    return np.array(lick_sensitivities), np.unique(trial_numbers)
    


def get_stop_selectivity(session):
    trial_numbers = np.array(session['trial_number'])
    position = np.array(session['P'])
    speed = np.array(session['S'])
    stop_mask = speed<3
    stop_mask = curate_behavioural_mask(stop_mask)

    # calc stop selectivity
    # 10 cm before rz is included for anticipitory action
    stop_selectivities = []
    for tn in np.unique(trial_numbers):
        trial_context = session['trials'][session['trials']['trial_number'] == tn]['trial_context'].values[0]
        if trial_context == 'rz1':
            rz_bounds = (80, 110)
        elif trial_context == 'rz2':
            rz_bounds = (110, 140)

        tn_mask = trial_numbers == tn
        rz_mask = ((position>rz_bounds[0]) & (position<rz_bounds[1]))
        elsewhere_mask = ((position<rz_bounds[0]) | (position>rz_bounds[1]))
        
        n_stops_in_rz =     len(position[(tn_mask & stop_mask & rz_mask)])
        n_stops_elsewhere = len(position[(tn_mask & stop_mask & elsewhere_mask)])
        stop_sensitivity = (n_stops_in_rz - n_stops_elsewhere)/(n_stops_in_rz + n_stops_elsewhere + 1e-10)

        stop_selectivities.append(stop_sensitivity)

    return np.array(stop_selectivities), np.unique(trial_numbers)
    
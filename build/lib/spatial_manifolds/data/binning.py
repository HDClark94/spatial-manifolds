import numpy as np


def get_bin_config(session_type):
    if 'OF' in session_type:
        return {
            ('P_x', 'P_y'): {
                'dim': '2d',
                'bounds': (0.0, 100.0, 0.0, 100.0),
                'num_bins': 40,
                'smooth_sigma': 1.5,
            },
            'S': {
                'dim': '1d',
                'bounds': (3.0, 45.0),
                'num_bins': 15,
            },
            'T': {
                'dim': '1dc',
                'bounds': (-np.pi, np.pi),
                'num_bins': 10,
            },
            'H': {
                'dim': '1dc',
                'bounds': (-np.pi, np.pi),
                'num_bins': 20,
            },
        }
    elif session_type == 'VR':
        return {
            'P': {
                'dim': '1d',
                'bounds': (0.0, 200.0),
                'num_bins': 100,
                'smooth_sigma': 2.5,
                'regions': {
                    (0, 1): {
                        'outbound': (30.0, 90.0),
                        'homebound': (110.0, 170.0),
                    },
                },
            },
            'S': {
                'dim': '1d',
                'bounds': (3.0, 90.0),
                'num_bins': 30,
            },
            'T': {
                'dim': '1dc',
                'bounds': (0.0, 2 * np.pi),
                'num_bins': 10,
            },
        }
    elif session_type == 'MCVR':
        return {
            'P': {
                'dim': '1d',
                'bounds': (0.0, 230.0),
                'num_bins': 100,
                'smooth_sigma': 2.5,
                'regions': {
                    (0, 1): {
                        'outbound': (30.0, 90.0),
                        'homebound': (110.0, 200.0),
                    },
                    (2, 3): {
                        'outbound': (30.0, 120.0),
                        'homebound': (140.0, 200.0),
                    },
                },
            },
            'S': {
                'dim': '1d',
                'bounds': (3.0, 90.0),
                'num_bins': 30,
            },
            'T': {
                'dim': '1dc',
                'bounds': (0.0, 2 * np.pi),
                'num_bins': 10,
            },
        }
    else:
        raise ValueError(f'Unknown session type: {session_type}')


def get_cont_vars(task):
    if 'of' in task:
        return ['P_x', 'P_y', 'S', 'H']
    else:
        return ['P', 'S']


def get_disc_vars(task):
    if 'of' in task:
        return []
    else:
        return ['trial_number', 'trial_type']


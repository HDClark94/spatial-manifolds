from .grid_score import compute_ellipse_grid_score, compute_grid_score
from .hd_mean_vector_length import compute_hd_mean_vector_length
from .ramp_class import compute_ramp_class
from .spatial_information import compute_spatial_information
from .speed_correlation import compute_speed_correlation
from .stability import compute_stability
from .theta_index import compute_theta_index
from .tuning_scores import compute_tuning_score

TUNING_SCORES = {
    'OF': {
        'ellipse_grid_score': (compute_ellipse_grid_score, True),
        'grid_score': (compute_grid_score, True),
        'hd_mean_vector_length': (compute_hd_mean_vector_length, True),
        'speed_correlation': (compute_speed_correlation, True),
        'theta_index': (compute_theta_index, False),
        'spatial_information': (compute_spatial_information, True),
    },
    'VR': {
        'ramp_class': (compute_ramp_class, True),
        'stability': (compute_stability, True),
        'theta_index': (compute_theta_index, False),
        'spatial_information': (compute_spatial_information, True),
        'speed_correlation': (compute_speed_correlation, True),
    },
    'MCVR': {
        'stability': (compute_stability, True),
        'ramp_class': (compute_ramp_class, True),
        'theta_index': (compute_theta_index, False),
        'spatial_information': (compute_spatial_information, True),
        'speed_correlation': (compute_speed_correlation, True),
    },
}

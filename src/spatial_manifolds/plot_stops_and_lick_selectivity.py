import pynapple as nap
from spatial_manifolds.toroidal import *
from spatial_manifolds.behaviour_plots import *

savepath = '/Users/harryclark/Documents/figs/grids'
mouses = [20,20,20,20,20,20,20,20,20,20,20,20,20,21,21,21,21,21,21,21,21,21,21,21,21,22,22,22,22,22,22,22,22,22,25,25,25,25,25,25,25,25,25,25,26,26,26,26,26,26,26,26,26,27,27,27,27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,28,29,29,29,29,29,29,29,29,29]
days =   [14,15,16,17,18,19,20,21,22,23,24,25,26,15,16,17,18,19,20,21,22,23,24,25,26,33,34,35,36,37,38,39,40,41,16,17,18,19,20,21,22,23,24,25,11,12,13,14,15,16,17,18,19,16,17,18,19,20,21,22,23,24,26,16,17,18,19,20,21,22,23,25,16,17,18,19,20,21,22,23,25]

mouses = [20,20,21,21,21,21,21,21,21,21,21,21,21,21,22,22,22,22,22,22,22,22,22,25,25,25,25,25,25,25,25,25,25,26,26,26,26,26,26,26,26,26,27,27,27,27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,28,29,29,29,29,29,29,29,29,29]
days =   [25,26,15,16,17,18,19,20,21,22,23,24,25,26,33,34,35,36,37,38,39,40,41,16,17,18,19,20,21,22,23,24,25,11,12,13,14,15,16,17,18,19,16,17,18,19,20,21,22,23,24,26,16,17,18,19,20,21,22,23,25,16,17,18,19,20,21,22,23,25]

for mouse, day in zip(mouses,days):
    print(f'mouse {mouse} day {day}')
    vr_folder = f'/Users/harryclark/Downloads/COHORT12/M{mouse}/D{day:02}/VR/'
    beh_path = vr_folder + f"sub-{mouse}_day-{day:02}_ses-VR_beh.nwb"
    beh = nap.load_file(beh_path, lazy_loading=False)

    plot_stops_and_sensitivity(beh, title=f'M{mouse}D{day}_stops', savepath=savepath)

"""Quick grid-score of a single weight snapshot: prints step, grid median, frac>0.3, max."""
import sys, os
REPO_DIR='/Users/harryclark/Documents/spatial-manifolds/GRID-PATTERN-FORMATION'
sys.path.insert(0, REPO_DIR)
import numpy as np, torch
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from scores import GridScorer
from visualize import compute_ratemaps

snap=sys.argv[1]; d=os.path.dirname(snap)
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
class O: pass
o=O(); o.Np=512;o.Ng=1024;o.sequence_length=20;o.batch_size=200
o.learning_rate=1e-4;o.weight_decay=1e-4;o.place_cell_rf=0.12;o.surround_scale=2
o.RNN_type='RNN';o.activation='relu';o.DoG=True;o.periodic=False
o.box_width=2.2;o.box_height=2.2;o.device=DEVICE
pc=PlaceCells(o); pc.us=torch.tensor(np.load(d+'/place_cells.npy'),dtype=torch.float32).to(DEVICE)
m=RNN(o,pc).to(DEVICE); m.load_state_dict(torch.load(snap,map_location=DEVICE)); m.eval()
tg=TrajectoryGenerator(o,pc)
a,_,_,_=compute_ratemaps(m,tg,o,res=40,n_avg=30,Ng=o.Ng)
scorer=GridScorer(40,((-1.1,1.1),(-1.1,1.1)),zip([0.2]*10,np.linspace(0.4,1.0,10).tolist()))
sc=np.array([scorer.get_scores(a[i])[0] for i in range(o.Ng)])
v=sc[~np.isnan(sc)]
step=os.path.basename(snap).replace('snap_','').replace('.pth','')
print(f"step {step}: grid_med={np.median(v):+.2f}  frac>0.3={(v>0.3).mean():.2f}  max={v.max():.2f}")

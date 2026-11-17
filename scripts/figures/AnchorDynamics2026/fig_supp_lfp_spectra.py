'''Figure S14: LFP spectra along every single shank, medial to lateral, per mouse.

One row per mouse, one column per shank (most medial left). Where a shank was
recorded on several days, the session with the most curated cells is shown.
Per channel (8 contacts, ~60 um): log Welch power, running only, z-scored per
frequency across the shank's channels (boundary_unsupervised.spectra; cached
in data/lfp/boundary/). Beside it, curated cells by depth and class.

Writes fig_supp_lfp_spectra.pdf
'''
import os, sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026')
import boundary_unsupervised as BU, fig7_supp_vis_examples as S, anatomy_rules as AR
from sklearn.decomposition import PCA
plt.rcParams['font.family']='Arial'
RC={'VIS':'#3f9b4f','MEC':'0.15','other':'0.65'}
CX=pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/cell_classifications_v2.csv')
CXc=CX.dropna(subset=['coord_SCs_x']).assign(ml=lambda d: d.coord_SCs_x.abs())
DC=pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/device_contact_id_annotations.csv')
SH=DC.groupby(['mouse','shank_id']).coord_SCs_x.mean().abs()
ss=sorted(S.single_shank_sessions())
# layout: per mouse, sessions placed in column = shank (medial -> lateral), row = repeat index
slots=[]   # (row_label, mouse, [4 x (day or None)])
for mo in sorted({m for m,_ in ss}):
    cols={k:[] for k in range(4)}
    for m_,dy in [s for s in ss if s[0]==mo]:
        ml=CXc[(CXc.mouse==mo)&(CXc.day==dy)].ml.median()
        k=int(np.argmin([abs(SH[(mo,j)]-ml) for j in range(4)])); cols[k].append(dy)
    def ncur(dy): return int(((CX.mouse==mo)&(CX.day==dy)&CX.cell_class_of1.isin(['GC','NGS','NS'])).sum())
    slots.append((mo,0,[max(cols[k],key=ncur) if cols[k] else None for k in range(4)]))
def load(mo,dy):
    cf=f'{BU.OUTD}/spectra_M{mo}D{dy}.npz'
    if os.path.exists(cf):
        z=np.load(cf,allow_pickle=True); return pd.DataFrame(z['C']),z['f'],z['LP']
    if not os.path.exists(f'{BU.LFP}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'): return None
    C,f,LP=BU.spectra(mo,dy); np.savez_compressed(cf,C=C.to_records(index=False),f=f,LP=LP); return C,f,LP
nr=len(slots)
fig=plt.figure(figsize=(10.4,1.9*nr))
G=fig.add_gridspec(nr,4,hspace=.55,wspace=.16,left=.06,right=.995,top=1-.35/nr,bottom=.75/nr)
for k in range(4):
    fig.text(.06+(k+.5)*.935/4, 1-.1/nr, f'shank {k+1} ' + ['(most medial)','','','(most lateral)'][k], ha='center', fontsize=8.5, weight='bold')
for r,(mo,rep,days) in enumerate(slots):
    for k,dy in enumerate(days):
        g=G[r,k].subgridspec(1,3,width_ratios=[.07,1,.48],wspace=.04)
        if dy is None:
            ax=fig.add_subplot(G[r,k]); ax.axis('off')
            ax.text(.5,.5,f'M{mo}: no single-shank session on shank {k+1}',ha='center',va='center',fontsize=8,color='0.5')
            continue
        cc=CX[(CX.mouse==mo)&(CX.day==dy)&CX.cell_class_of1.isin(['GC','NGS','NS'])].dropna(subset=['coord_SCs_y'])
        got=load(mo,dy)
        if got is None:
            ax=fig.add_subplot(G[r,k]); ax.axis('off')
            nv=cc.brain_region_atlas.astype(str).str.startswith('VIS').sum(); nm=cc.brain_region_atlas.astype(str).str.startswith('ENTm').sum()
            ax.text(.5,.5,f'M{mo} D{dy}  (shank {k+1}, {SH[(mo,k)]:.0f} um)\nno LFP file\ncurated cells: {nv} VIS, {nm} MEC',ha='center',va='center',fontsize=8,color='0.35')
            continue
        C,f,LP=got
        Z=(LP-LP.mean(0))/LP.std(0)
        has=(C.region=='VIS').any() and (C.region=='MEC').any()
        b=(C.dv[C.region=='VIS'].max()+C.dv[C.region=='MEC'].min())/2 if has else np.nan
        rule=AR.session_rule(mo,dy); sp=rule['boundary']; ex=rule['exit']
        ylim=(C.dv.max()+60,C.dv.min()-60)
        a0=fig.add_subplot(g[0])
        for d,rg in zip(C.dv,C.region): a0.barh(d,1,height=58,color=RC[rg])
        a0.set_ylim(ylim); a0.set_xticks([]); a0.tick_params(labelsize=6)
        if k==0: a0.set_ylabel(f'M{mo}\nDV (um)',fontsize=7)
        ax=fig.add_subplot(g[1])
        ax.imshow(Z,aspect='auto',cmap='RdBu_r',vmin=-2.5,vmax=2.5,extent=[0,len(f),ylim[0],ylim[1]],interpolation='nearest')
        xt=[10,60,200]; ax.set_xticks([np.argmin(abs(f-x)) for x in xt]); ax.set_xticklabels(xt,fontsize=5.5)
        ax.tick_params(labelleft=False); ax.set_xlabel('Hz',fontsize=5.5,labelpad=0)
        a2=fig.add_subplot(g[2],sharey=ax)
        rng=np.random.default_rng(0)
        colj=AR.cell_column(cc)
        for j in range(4):
            m=cc[colj==j]
            a2.scatter(j+rng.uniform(-.3,.3,len(m)),m.coord_SCs_y,s=AR.rate_size([AR.vr_rates(mo,dy).get(int(c_),.1) for c_ in m.cluster_id]),c=[RC[BU.region(x)] for x in m.brain_region_atlas],lw=0,alpha=.7)
        a2.set_xticks([0,1,2,3]); a2.set_xticklabels(['G','N','S','I'],fontsize=5); a2.set_xlim(-.6,3.6)
        a2.tick_params(labelleft=False,labelsize=5.5); a2.spines[['top','right']].set_visible(False)
        for a in (a0,ax,a2):
            if has: a.axhline(b,color='k',ls='--',lw=.8)
        for a in (ax,a2):
            a.axhline(sp,color='#e66101',lw=1.2)
            a.axhline(ex,color='#5e3c99',lw=1.0)
            a.axhspan(ex,ylim[0],color='0.6',alpha=.35,lw=0,zorder=3)
        ttl=f'D{dy}, {SH[(mo,k)]:.0f} um' + (f': {sp-b:+.0f} um vs atlas' if has and np.isfinite(sp) else '')
        ax.set_title(ttl,fontsize=5.6,loc='left',pad=2); a2.set_title(f'{len(cc)} cells',fontsize=5,pad=2)
# ---- key along the bottom
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
_y = .2 / nr
cax = fig.add_axes([.07, _y + .012, .17, .006])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-2.5, 2.5), cmap='RdBu_r'),
                  cax=cax, orientation='horizontal', ticks=[-2.5, 0, 2.5])
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6, length=2, pad=1)
cb.set_label('log power, z across the shank\'s channels (per frequency)', fontsize=6.5, labelpad=2)
h = [Line2D([], [], color='#e66101', lw=1.4, label='VIS - MEC/PARA boundary'),
     Line2D([], [], color='#5e3c99', lw=1.2, label='leaves cortex'),
     Patch(facecolor='0.6', alpha=.35, lw=0, label='below cortex'),
     Line2D([], [], color='k', ls='--', lw=.9, label='atlas VIS - MEC border'),
     Patch(facecolor=RC['VIS'], label='visual cortex'), Patch(facecolor=RC['MEC'], label='MEC'),
     Patch(facecolor=RC['other'], label='other (atlas)'),
     Line2D([], [], marker='o', ls='', color='0.4', ms=float(np.sqrt(AR.rate_size([5])[0])), label='5 Hz'),
     Line2D([], [], marker='o', ls='', color='0.4', ms=float(np.sqrt(AR.rate_size([50])[0])), label='50 Hz'),
     Line2D([], [], marker='o', ls='', color='0.4', ms=float(np.sqrt(AR.rate_size([100])[0])), label='100 Hz')]
fig.legend(handles=h, loc='lower left', bbox_to_anchor=(.27, _y - .004), ncol=5, fontsize=6.3, frameon=False,
           handlelength=1.6, columnspacing=1.1, title='lines, region colours (strip and cells), cell size = firing rate',
           title_fontsize=6.3)
OUT=f'{os.path.dirname(os.path.abspath(BU.__file__))}/fig_supp_lfp_spectra.pdf'
fig.savefig(OUT,dpi=200,bbox_inches='tight'); print('wrote',OUT)
print(nr,'rows')

bs = 2 
tl=200
vr_tl = 200
mcvr_tl = 230
time_bs = 100 # ms
time_bs = 10 # ms
 
rm_figsize = (1, 1.6)   
 
other_areas = [
    # Vermal lobules
    'LING', 'CUL2_3', 'CUL4, 5', 'DEC', 'FOTU', 'PYR', 'UVU', 'NOD',
    # Hemispheric lobules
    'SIM', 'ANcr1', 'ANcr2', 'FL', 'PFL', 'COPY',
    # Deep cerebellar nuclei
    'DN', 'IP', 'FN',
    # Cerebellar white matter / peduncles
    'arb', 'mcp', 'scp', 'icp',
    # Vestibular nuclei
    'VCO', 'LAV', 'MV', 'SPIV', 'SUV',
    # Out of brain areas
    'root',
    # medulla, pons, cerebellum parent, 
    'MY', 'P', 'CB',
]

disqualifying_brain_areas_for_grid_cells = ['dhc', 'FL', 'PFL', 'SIM', 'arb', 'root', 'mcp', 'VIS', 'RSPagl6a',
                                            'RSPd5', 'RSPd6a', 'VISl1', 'VISl2/3', 'VISp1', 'VISp2/3', 
                                            'VISp4','VISp5', 'VISp6a', 'VISp6b', 'VISpl1', 'VISpl2/3', 'VISpl4', 
                                            'VISpl5', 'VISpl6a', 'VISpor5', 'mcp', 'root', 'APr', 'SUB', 'POST',
                                            'CUL4, 5','VCO', 'VISl4', 'VISl5', 'VISl6a', 'VISl6b', 'VISp6b', 
                                            'VISpl6b', 'VISpor6a', 'VISpor6b', 'dhc', 'ec', 'fp', 'or']